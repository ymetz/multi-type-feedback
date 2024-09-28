"""Module for training a reward model from the generated feedback."""

import argparse
import math
import os
import pickle
from os import path
from pathlib import Path
from random import randint, randrange
from typing import Union, List, Tuple
from numpy.typing import NDArray
import gymnasium as gym
import random
from scipy.stats import truncnorm, uniform

import numpy as np
import torch
from pytorch_lightning import Callback, LightningModule, Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import DataLoader, Dataset, random_split

from gymnasium.wrappers.stateful_observation import FrameStackObservation
from gymnasium.wrappers.transform_observation import TransformObservation
from minigrid.wrappers import FlatObsWrapper 
from procgen import ProcgenGym3Env
from rl_zoo3.wrappers import Gym3ToGymnasium
from stable_baselines3.common.vec_env import VecExtractDictObs
from stable_baselines3.common.atari_wrappers import AtariWrapper
import ale_py
import minigrid

import wandb
from rlhf.datatypes import FeedbackDataset, FeedbackType, SegmentT
from rlhf.networks import LightningNetwork, LightningCnnNetwork, calculate_single_reward_loss, calculate_pairwise_loss

script_path = Path(__file__).parents[1].resolve()

# Utilize Tensor Cores of NVIDIA GPUs
torch.set_float32_matmul_precision("high")

def truncated_uniform_vectorized(mean, width, low=0, upp=9):
    # Handle scalar inputs
    scalar_input = np.isscalar(mean) and np.isscalar(width)
    mean = np.atleast_1d(mean)
    width = np.atleast_1d(width)
    
    # Calculate the bounds of the uniform distribution
    lower = mean - width/2
    upper = mean + width/2
    
    # Clip the bounds to [low, upp]
    lower = np.clip(lower, low, upp)
    upper = np.clip(upper, low, upp)
    
    # Generate random values
    r = np.random.uniform(size=mean.shape)
    
    # Calculate the result
    result = lower + r * (upper - lower)
    
    return result[0] if scalar_input else result

class FeedbackDataset(Dataset):
    """PyTorch Dataset for loading the feedback data."""

    def __init__(
        self,
        dataset_path: str,
        feedback_type: FeedbackType,
        n_feedback: int,
        noise_level: float = 0.0, # depending on the feedback, we use different types of noise (e.g. flip the preference, add noise to the rating or description)
        segment_len: int = 50,
        env = None,
    ):
        """Initialize dataset."""
        print("Loading dataset...")

        self.targets: Union[List[SegmentT], List[NDArray], Tuple[SegmentT, SegmentT], Tuple[NDArray, NDArray]] = []
        self.preds: List[int] = []

        with open(dataset_path, "rb") as feedback_file:
            feedback_data: FeedbackDataset = pickle.load(feedback_file)

        if feedback_type == "evaluative":
            for seg in feedback_data["segments"]:
                obs = torch.vstack([torch.as_tensor(p[0]).float() for p in seg])
                actions = torch.vstack([torch.as_tensor(p[1]).float() for p in seg])

                # Pad both trajectories to the maximum length
                len_obs = obs.size(0)

                if len_obs < segment_len:
                    pad_size = segment_len - len_obs
                    obs = torch.cat([obs, torch.zeros(pad_size, obs.size(1))], dim=0)
                    actions = torch.cat([actions, torch.zeros(pad_size, actions.size(1))], dim=0)
                
                self.targets.append((obs, actions))
                self.preds = feedback_data["ratings"]
                # add noise to the ratings
                if noise_level > 0:
                    # apply noise, accounting for clipping at the borders [0,9]
                    self.preds = truncated_uniform_vectorized(
                        mean=self.preds, 
                        width=noise_level*10*np.ones_like(self.preds), 
                        low=0, 
                        upp=9
                    )
        elif feedback_type == "comparative":
            for comp in feedback_data["preferences"]:
                # seg 1
                obs = torch.vstack([torch.as_tensor(p[0]).float() for p in feedback_data["segments"][comp[0]]])
                actions = torch.vstack([torch.as_tensor(p[1]).float() for p in feedback_data["segments"][comp[0]]])

                # seg 2
                obs2 = torch.vstack([torch.as_tensor(p[0]).float() for p in feedback_data["segments"][comp[1]]])
                actions2 = torch.vstack([torch.as_tensor(p[1]).float() for p in feedback_data["segments"][comp[1]]])

                # Pad both trajectories to the maximum length, necessary for batching with data loader
                len_obs = obs.size(0)
                len_obs2 = obs2.size(0)

                if len_obs < segment_len:
                    pad_size = segment_len - len_obs
                    obs = torch.cat([obs, torch.zeros(pad_size, obs.size(1))], dim=0)
                    actions = torch.cat([actions, torch.zeros(pad_size, actions.size(1))], dim=0)
                if len_obs2 < segment_len:
                    pad_size = segment_len - len_obs2
                    obs2 = torch.cat([obs2, torch.zeros(pad_size, obs2.size(1))], dim=0)
                    actions2 = torch.cat([actions2, torch.zeros(pad_size, actions2.size(1))], dim=0)
                
                # flip the preference with a certain probability
                if random.random() < noise_level:
                    self.targets.append(((obs2, actions2),(obs, actions)))
                    self.preds.append(comp[2])
                else:
                    self.targets.append(((obs, actions),(obs2, actions2)))
                    self.preds.append(comp[2])
                    
        elif feedback_type == "demonstrative":
            for demo in feedback_data["demos"]:
                obs = torch.vstack([torch.as_tensor(p[0]).float() for p in demo])
                actions = torch.vstack([torch.as_tensor(p[1]).float() for p in demo])

                if noise_level > 0.0:
                    # Get the shape of the actions tensor
                    num_actions, action_dim = actions.shape
                    
                    # Calculate how many actions to replace
                    num_replace = int(num_actions * noise_level)
                    
                    # Randomly select indices to replace
                    replace_indices = np.random.choice(num_actions, num_replace, replace=False)
                    
                    # Create a new tensor to store the modified actions
                    new_actions = actions.clone()
                    
                    # Replace selected actions with random actions from the environment
                    for idx in replace_indices:
                        random_action = env.action_space.sample()
                        new_actions[idx] = torch.as_tensor(random_action).float()

                    actions = new_actions

                # just use a random segment as the opposite
                rand_index = random.randrange(0, len(feedback_data["segments"]))
                obs_rand = torch.vstack([torch.as_tensor(p[0]).float() for p in feedback_data["segments"][rand_index]])
                actions_rand = torch.vstack([torch.as_tensor(p[1]).float() for p in feedback_data["segments"][rand_index]])

                # Pad both trajectories to the maximum length
                len_obs = obs.size(0)
                len_obs_rand = obs_rand.size(0)

                if len_obs < segment_len:
                    pad_size = segment_len - len_obs
                    obs = torch.cat([obs, torch.zeros(pad_size, obs.size(1))], dim=0)
                    actions = torch.cat([actions, torch.zeros(pad_size, actions.size(1))], dim=0)
                if len_obs_rand < segment_len:
                    pad_size = segment_len - len_obs_rand
                    obs_rand = torch.cat([obs_rand, torch.zeros(pad_size, obs_rand.size(1))], dim=0)
                    actions_rand = torch.cat([actions_rand, torch.zeros(pad_size, actions_rand.size(1))], dim=0)
                
                self.targets.append(((obs_rand, actions_rand), (obs, actions)))
                self.preds.append(1) # assume that the demonstration is optimal, maybe add confidence value (based on regret)
        elif feedback_type == "corrective":
            for comp in feedback_data["corrections"]:
                obs = torch.vstack([torch.as_tensor(p[0]).float() for p in comp[0]])
                actions = torch.vstack([torch.as_tensor(p[1]).float() for p in comp[0]])

                obs2 = torch.vstack([torch.as_tensor(p[0]).float() for p in comp[1]])
                actions2 = torch.vstack([torch.as_tensor(p[1]).float() for p in comp[1]])

                # Pad both trajectories to the maximum length
                len_obs = obs.size(0)
                len_obs2 = obs2.size(0)

                if len_obs < segment_len:
                    pad_size = segment_len - len_obs
                    obs = torch.cat([obs, torch.zeros(pad_size, obs.size(1))], dim=0)
                    actions = torch.cat([actions, torch.zeros(pad_size, actions.size(1))], dim=0)
                if len_obs2 < segment_len:
                    pad_size = segment_len - len_obs2
                    obs2 = torch.cat([obs2, torch.zeros(pad_size, obs2.size(1))], dim=0)
                    actions2 = torch.cat([actions2, torch.zeros(pad_size, actions2.size(1))], dim=0)
            
                # flip the preference with a certain probability
                if random.random() < noise_level:
                    self.targets.append(((obs2, actions2),(obs, actions)))
                    self.preds.append(1) 
                else:
                    self.targets.append(((obs, actions),(obs2, actions2)))
                    self.preds.append(1) 
        elif feedback_type == "descriptive":

            rew_range = np.array([cr[2] for cr in feedback_data["description"]]).var()
            
            for cluster_representative in feedback_data["description"]:
                self.targets.append((torch.as_tensor(cluster_representative[0]).unsqueeze(0).float(), torch.as_tensor(cluster_representative[1]).unsqueeze(0).float()))
                
                if noise_level > 0.0:
                    rew = cluster_representative[2] + np.random.uniform(-noise_level*rew_range, noise_level*rew_range)
                    self.preds.append(rew)
                else:
                    self.preds.append(cluster_representative[2])
        elif feedback_type == "cluster_preferences":
            for cluster_representative in feedback_data["cluster_description"]:
                self.targets.append((np.expand_dims(cluster_representative[0], 0), np.expand_dims(cluster_representative[1], 0)))
                if noise_level > 0.0:
                    cluster_variance = 0
                    rew = cluster_representative[2] + np.random.uniform(-noise_level*10, noise_level*10)
                    self.preds.append(rew)
                else:
                    self.preds.append(cluster_representative[2])
        elif feedback_type == "descriptive_preference":
            for cpref in feedback_data["description_preference"]:
                idx_1 = cpref[0]
                                
                # cluster 1
                obs = torch.as_tensor(feedback_data["description"][idx_1][0]).unsqueeze(0).float()
                actions = torch.as_tensor(feedback_data["description"][idx_1][1]).unsqueeze(0).float()

                idx_2 = cpref[1]
                
                # cluster 2
                obs2 = torch.as_tensor(feedback_data["description"][idx_2][0]).unsqueeze(0).float()
                actions2 = torch.as_tensor(feedback_data["description"][idx_2][1]).unsqueeze(0).float()

                # flip the preference with a certain probability
                if random.random() < noise_level:
                    self.targets.append(((obs2, actions2),(obs, actions)))
                    self.preds.append(cpref[2])
                else:
                    self.targets.append(((obs, actions),(obs2, actions2)))
                    self.preds.append(cpref[2])
        else:
            raise NotImplementedError(
                "Dataset not implemented for this feedback type."
            )

        print("Dataset loaded")

    def __len__(self):
        """Return size of dataset."""
        return len(self.targets)

    def __getitem__(self, index):
        """Return item with given index."""
        return self.targets[index], self.preds[index]


def train_reward_model(
    reward_model: LightningModule,
    reward_model_id: str,
    feedback_type: FeedbackType,
    dataset: FeedbackDataset,
    maximum_epochs: int = 100,
    cpu_count: int = 4,
    algorithm: str = "sac",
    environment: str = "HalfCheetah-v3",
    gradient_clip_value: Union[float, None] = None,
    split_ratio: float = 0.8,
    enable_progress_bar=True,
    callback: Union[Callback, None] = None,
    num_ensemble_models: int = 4,
    noise_level: float = 0.0,
    seed: int = 0,
):

    """Train a reward model given trajectories data."""
    training_set_size = math.floor(split_ratio * len(dataset))
    train_set, val_set = random_split(
        dataset, lengths=[training_set_size, len(dataset) - training_set_size]
    )

    train_loader = DataLoader(
        train_set,
        batch_size=num_ensemble_models,
        shuffle=True,
        pin_memory=True,
        #num_workers=cpu_count,
        num_workers=0,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_set, 
        batch_size=num_ensemble_models, 
        pin_memory=True, 
        num_workers=1, 
        drop_last=True
    )

    checkpoint_callback = ModelCheckpoint(
        dirpath=path.join(script_path, "reward_models_2"),
        filename=reward_model_id,
        monitor="val_loss",
    )

    # initialise the wandb logger and name your wandb project
    # initialise the wandb logger and name your wandb project
    wandb_logger = WandbLogger(project="multi_reward_feedback_final", 
                               name=reward_model_id,
                               config={
                                    "feedback_type": feedback_type,
                                    "noise_level": noise_level,
                                    "seed": seed,
                                    "environment": environment,
                                },
    )

    trainer = Trainer(
        max_epochs=maximum_epochs,
        devices=[0],
        log_every_n_steps=5,
        gradient_clip_val=gradient_clip_value,
        enable_progress_bar=enable_progress_bar,
        logger=wandb_logger,
        accumulate_grad_batches=32,
        callbacks=[
            EarlyStopping(monitor="val_loss", mode="min", patience=5),
            checkpoint_callback,
            *([callback] if callback is not None else []),
        ],
    )

    trainer.fit(reward_model, train_loader, val_loader)

    wandb.finish()

    return reward_model


def main():

    cpu_count = os.cpu_count()
    cpu_count = cpu_count if cpu_count is not None else 8

    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument(
        "--feedback-type",
        type=str,
        default="evaluative",
        help="Type of feedback to train the reward model",
    )
    arg_parser.add_argument(
        "--algorithm",
        type=str,
        default="ppo",
        help="RL algorithm used to generate the feedback",
    )
    arg_parser.add_argument(
        "--environment",
        type=str,
        default="HalfCheetah-v5",
        help="Environment used to generate the feedback",
    )
    arg_parser.add_argument(
        "--seed",
        type=int,
        default=12,
        help="The seed for random generation adn the saved feedback",
    )
    arg_parser.add_argument(
        "--n-ensemble",
        type=int,
        default=4,
    )
    arg_parser.add_argument(
        "--n-feedback",
        type=int,
        default=-1,
    )
    arg_parser.add_argument(
        "--noise-level",
        type=float,
        default=0.0,
        help="Noise level to add to the feedback",
    )
    arg_parser.add_argument(
        '--no-loading-bar', 
        action='store_true', 
        help="Disable the loading bar"
    )
    args = arg_parser.parse_args()

    FEEDBACK_ID = "_".join(
        [args.algorithm, args.environment, str(args.seed)]
    )
    if args.noise_level > 0.0:
        MODEL_ID = f"{FEEDBACK_ID}_{args.feedback_type}_{args.seed}_noise_{str(args.noise_level)}"
    else:
        MODEL_ID = f"{FEEDBACK_ID}_{args.feedback_type}_{args.seed}"

    # Select loss function based on feedback type
    loss_function = None
    architecture_cls = None

    if args.feedback_type == "evaluative" or args.feedback_type == "descriptive":
        loss_function = calculate_single_reward_loss
    else:
        #"comparative" | "corrective" | "demonstrative" | "descriptive_preference":
        loss_function = calculate_pairwise_loss

    if "procgen" in args.environment:
        _, short_name, _ = args.environment.split("-")
        environment = Gym3ToGymnasium(ProcgenGym3Env(num=1, env_name=short_name))
        environment = TransformObservation(environment, lambda obs: obs["rgb"], environment.observation_space)
    elif "ALE/" in args.environment:
        environment = FrameStackObservation(AtariWrapper(gym.make(args.environment)), 4)
        environment = TransformObservation(environment, lambda obs: obs.squeeze(-1), environment.observation_space)
    elif "MiniGrid" in args.environment:
        environment = FlatObsWrapper(gym.make(args.environment))
    else:
        environment = gym.make(args.environment)
    
    if "procgen" in args.environment or "ALE" in args.environment:
        reward_model = LightningCnnNetwork(
            input_spaces=(environment.observation_space, environment.action_space),
            hidden_dim=256,
            action_hidden_dim=16,
            layer_num=3,
            cnn_channels=(16,32,32),
            output_dim=1,
            loss_function=loss_function,
            learning_rate=(
                1e-5
                #1e-6
                #if args.feedback_type == "corrective"
                #else (1e-5 if args.feedback_type == "comparative" else 2e-5)
            ),
            ensemble_count=args.n_ensemble,
        )

    else:
        reward_model = LightningNetwork(
            input_spaces=(environment.observation_space, environment.action_space),
            hidden_dim=256,
            action_hidden_dim=32,
            layer_num=6,
            output_dim=1,
            loss_function=loss_function,
            learning_rate=(
                1e-5
                #1e-6
                #if args.feedback_type == "corrective"
                #else (1e-5 if args.feedback_type == "comparative" else 2e-5)
            ),
            ensemble_count=args.n_ensemble,
        )

    # Load data
    dataset_dir = "feedback_regen"
    dataset = FeedbackDataset(
        path.join(script_path, dataset_dir, f"{FEEDBACK_ID}.pkl"),
        args.feedback_type,
        args.n_feedback,
        noise_level=args.noise_level,
        env=environment if args.feedback_type == "demonstrative" else None,
    )

    train_reward_model(
        reward_model,
        MODEL_ID,
        args.feedback_type,
        dataset,
        maximum_epochs=100,
        split_ratio=0.85,
        cpu_count=cpu_count,
        num_ensemble_models=args.n_ensemble,
        enable_progress_bar=False if args.no_loading_bar else True,
        noise_level=args.noise_level,
        seed=args.seed,
    )



if __name__ == "__main__":
    main()
