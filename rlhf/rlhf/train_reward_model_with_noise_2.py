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
import highway_env
from rl_zoo3.wrappers import Gym3ToGymnasium
from stable_baselines3.common.vec_env import VecExtractDictObs
from stable_baselines3.common.atari_wrappers import AtariWrapper
import ale_py
import minigrid
from rl_zoo3.utils import ppo_make_metaworld_env

import wandb
from rlhf.datatypes import FeedbackDataset, FeedbackType, SegmentT
from rlhf.networks import LightningNetwork, LightningCnnNetwork, calculate_single_reward_loss, calculate_pairwise_loss

# for convenice sake, todo: make dynamic in the future
discount_factors = {
    'HalfCheetah-v5': 0.98,
    'Hopper-v5': 0.99,
    'Swimmer-v5': 0.9999,
    'Ant-v5': 0.99,
    'Walker2d-v5': 0.99,
    'ALE/BeamRider-v5': 0.99,
    'ALE/MsPacman-v5': 0.99,
    'ALE/Enduro-v5': 0.99,
    'ALE/Pong-v5': 0.99,
    'Humanoid-v5': 0.99,
    'highway-v0': 0.99,
    'merge-v0': 0.99,
    'roundabout-v0': 0.99,
    'metaworld-sweep-into-v2': 0.99,
    'metaworld-button-press-v2': 0.99,
    'metaworld-pick-place-v2': 0.99
}

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

def truncated_gaussian_vectorized(mean, width, low=0, upp=9, min_width=1e-8):
    # Handle scalar inputs
    scalar_input = np.isscalar(mean) and np.isscalar(width)
    mean = np.atleast_1d(mean)
    width = np.atleast_1d(width)
    
    # Calculate the bounds of the distribution
    lower = np.maximum(mean - width/2, low)
    upper = np.minimum(mean + width/2, upp)
    
    # Calculate parameters for truncated normal distribution
    a = (lower - mean) / (width/4)  # 4 sigma range
    b = (upper - mean) / (width/4)

    print(a,b)
    
    # Generate samples from truncated normal distribution
    result = truncnorm.rvs(a, b, loc=mean, scale=width/4, size=mean.shape)
    
    return result[0] if scalar_input else result

def discounted_sum_numpy(rewards, gamma):
    return np.sum(rewards * (gamma ** np.arange(len(rewards))))

class FeedbackDataset(Dataset):
    """PyTorch Dataset for loading the feedback data."""

    def __init__(
        self,
        dataset_path: str,
        feedback_type: FeedbackType,
        n_feedback: int,
        env_name: str = "",
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
                    obs = torch.cat([obs, torch.zeros(pad_size, *obs.shape[1:])], dim=0)
                    actions = torch.cat([actions, torch.zeros(pad_size, *actions.shape[1:])], dim=0)
                
                self.targets.append((obs, actions))
            
            self.preds = feedback_data["ratings"]
            # add noise to the ratings
            if noise_level > 0:
                # apply noise, accounting for clipping at the borders [0,9]
                self.preds = truncated_gaussian_vectorized(
                    mean=self.preds, 
                    width=noise_level*10*np.ones_like(self.preds), 
                    low=0, 
                    upp=9
                )
        elif feedback_type == "comparative":
            
            rews_min, rews_max = np.min([e * -1 for e in feedback_data["opt_gaps"]]), np.max([e * -1 for e in feedback_data["opt_gaps"]])
            ref_diff = np.abs(rews_max - rews_min)

            flipped = 0
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
                    obs = torch.cat([obs, torch.zeros(pad_size, *obs.shape[1:])], dim=0)
                    actions = torch.cat([actions, torch.zeros(pad_size, *actions.shape[1:])], dim=0)
                if len_obs2 < segment_len:
                    pad_size = segment_len - len_obs2
                    obs2 = torch.cat([obs2, torch.zeros(pad_size, *obs2.shape[1:])], dim=0)
                    actions2 = torch.cat([actions2, torch.zeros(pad_size, *actions2.shape[1:])], dim=0)
                
                # add noise and recompute preferences
                if noise_level > 0:
                    rew1 = -feedback_data["opt_gaps"][comp[0]]
                    rew2 = -feedback_data["opt_gaps"][comp[1]]

                    rew1 = truncated_gaussian_vectorized(
                        mean=np.array(rew1), 
                        width=np.array(noise_level) * ref_diff, 
                        low=rews_min,
                        upp=rews_max
                    )
                    rew2 = truncated_gaussian_vectorized(
                        mean=np.array(rew2), 
                        width=np.array(noise_level) * ref_diff, 
                        low=rews_min,
                        upp=rews_max
                    ) 

                    if rew2 > rew1:
                        self.targets.append(((obs, actions),(obs2, actions2)))
                    else:
                        self.targets.append(((obs2, actions2),(obs, actions)))
                        flipped += 1
                    self.preds.append(comp[2])
                else:
                    self.targets.append(((obs, actions),(obs2, actions2)))
                    self.preds.append(comp[2])

        elif feedback_type == "demonstrative":

            with open(os.path.join("samples", f"random_{env_name}.pkl"), "rb") as random_file:
                random_data = pickle.load(random_file)
            
            for demo in feedback_data["demos"]:
                obs = np.vstack([p[0] for p in demo])
                actions = np.vstack([p[1] for p in demo])

                if noise_level > 0.0:

                    obs_min, obs_max = np.min(obs, axis=0), np.max(obs, axis=0)
                    obs_std = np.std(obs, axis=0)
                    acts_min, acts_max = np.min(actions, axis=0), np.max(actions, axis=0)
                    acts_std = np.std(actions, axis=0)
                    
                    obs = truncated_gaussian_vectorized(
                        mean=obs, 
                        width=np.array(noise_level) * obs_std, 
                        low=obs_min,
                        upp=obs_max,
                    )
                    
                    actions = truncated_gaussian_vectorized(
                        mean=actions, 
                        width=np.array(noise_level) * acts_std, 
                        low=acts_min,
                        upp=acts_max,
                    )

                obs = torch.as_tensor(obs).float()
                actions = torch.as_tensor(actions).float()

                # just use a random segment as the opposite
                rand_index = random.randrange(0, len(random_data["segments"]))
                obs_rand = torch.vstack([torch.as_tensor(p[0]).float() for p in random_data["segments"][rand_index]])
                actions_rand = torch.vstack([torch.as_tensor(p[1]).float() for p in random_data["segments"][rand_index]])

                # Pad both trajectories to the maximum length
                len_obs = obs.size(0)
                len_obs_rand = obs_rand.size(0)

                if len_obs < segment_len:
                    pad_size = segment_len - len_obs
                    obs = torch.cat([obs, torch.zeros(pad_size, *obs.shape[1:])], dim=0)
                    actions = torch.cat([actions, torch.zeros(pad_size, *actions.shape[1:])], dim=0)
                if len_obs_rand < segment_len:
                    pad_size = segment_len - len_obs_rand
                    obs_rand = torch.cat([obs_rand, torch.zeros(pad_size, *obs_rand.shape[1:])], dim=0)
                    actions_rand = torch.cat([actions_rand, torch.zeros(pad_size, *actions_rand.shape[1:])], dim=0)
                
                self.targets.append(((obs_rand, actions_rand), (obs, actions)))
                self.preds.append(1) # assume that the demonstration is optimal, maybe add confidence value (based on regret)
        elif feedback_type == "corrective":

            rews_min, rews_max = np.min([e * -1 for e in feedback_data["opt_gaps"]]), np.max([e * -1 for e in feedback_data["opt_gaps"]])
            rew_diff = np.abs(rews_max - rews_min)
            gamma = discount_factors[env_name]
            
            flipped = 0
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
                    obs = torch.cat([obs, torch.zeros(pad_size, *obs.shape[1:])], dim=0)
                    actions = torch.cat([actions, torch.zeros(pad_size, *actions.shape[1:])], dim=0)
                if len_obs2 < segment_len:
                    pad_size = segment_len - len_obs2
                    obs2 = torch.cat([obs2, torch.zeros(pad_size, *obs2.shape[1:])], dim=0)
                    actions2 = torch.cat([actions2, torch.zeros(pad_size, *actions2.shape[1:])], dim=0)
            
                # add noise and recompute preferences
                if noise_level > 0.0:
                    rews1 = discounted_sum_numpy(np.array([p[2] for p in comp[0]]), gamma)
                    rews2 = discounted_sum_numpy(np.array([p[2] for p in comp[1]]), gamma)

                    rew1 = truncated_gaussian_vectorized(
                        mean=rews1, 
                        width=np.array(noise_level) * rew_diff, 
                        low=min(rews_min, rews1),
                        upp=max(rews_max, rews1),
                    ).item()
                    rew2 = truncated_gaussian_vectorized(
                        mean=rews2, 
                        width=np.array(noise_level) * rew_diff, 
                        low=min(rews_min, rews2),
                        upp=max(rews_max, rews2)
                    ).item()

                    if rew2 > rew1:
                        self.targets.append(((obs, actions),(obs2, actions2)))
                    else:
                        self.targets.append(((obs2, actions2),(obs, actions)))
                        flipped += 1
                    self.preds.append(1)
                else:
                    self.targets.append(((obs, actions),(obs2, actions2)))
                    self.preds.append(1)
        elif feedback_type == "descriptive":

            cluster_rews = np.array([cr[2] for cr in feedback_data["description"]])
            cluster_rew_min, cluster_rew_max = cluster_rews.min(), cluster_rews.max()
            cluster_rew_diff = np.abs(cluster_rew_max - cluster_rew_min)
            
            for cluster_representative in feedback_data["description"]:
                self.targets.append((torch.as_tensor(cluster_representative[0]).unsqueeze(0).float(), torch.as_tensor(cluster_representative[1]).unsqueeze(0).float()))
                
                if noise_level > 0.0:
                    rew = truncated_gaussian_vectorized(
                        mean=np.array(cluster_representative[2]), 
                        width=np.array(noise_level) * cluster_rew_diff, 
                        low=cluster_rew_min,
                        upp=cluster_rew_max
                    ) 
                    self.preds.append(rew.item())
                else:
                    self.preds.append(cluster_representative[2])
        elif feedback_type == "descriptive_preference":

            cluster_rews = np.array([cr[2] for cr in feedback_data["description"]])
            cluster_rew_min, cluster_rew_max = cluster_rews.min(), cluster_rews.max()
            cluster_rew_diff = np.abs(cluster_rew_max - cluster_rew_min)
            
            flipped = 0
            for cpref in feedback_data["description_preference"]:
                idx_1 = cpref[0]
                                
                # cluster 1
                obs = torch.as_tensor(feedback_data["description"][idx_1][0]).unsqueeze(0).float()
                actions = torch.as_tensor(feedback_data["description"][idx_1][1]).unsqueeze(0).float()

                idx_2 = cpref[1]
                
                # cluster 2
                obs2 = torch.as_tensor(feedback_data["description"][idx_2][0]).unsqueeze(0).float()
                actions2 = torch.as_tensor(feedback_data["description"][idx_2][1]).unsqueeze(0).float()

                # add noise and recompute preferences
                if noise_level > 0:
                    rew1 = feedback_data["description"][idx_1][2]
                    rew2 = feedback_data["description"][idx_2][2]

                    rew1 = truncated_gaussian_vectorized(
                        mean=np.array(rew1), 
                        width=np.array(noise_level) * cluster_rew_diff, 
                        low=cluster_rew_min,
                        upp=cluster_rew_max
                    ).item()
                    rew2 = truncated_gaussian_vectorized(
                        mean=np.array(rew2), 
                        width=np.array(noise_level) * cluster_rew_diff, 
                        low=cluster_rew_min,
                        upp=cluster_rew_max
                    ).item()

                    if rew2 > rew1:
                        self.targets.append(((obs, actions),(obs2, actions2)))
                    else:
                        self.targets.append(((obs2, actions2),(obs, actions)))
                        flipped += 1
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
        dirpath=path.join(script_path, "reward_models_lul"),
        filename=reward_model_id,
        monitor="val_loss",
    )

    # initialise the wandb logger and name your wandb project
    # initialise the wandb logger and name your wandb project
    wandb_logger = WandbLogger(project="multi_reward_feedback_final_lul", 
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

    np.random.seed(args.seed)
    random.seed(args.seed)

    env_name = args.environment if "ALE" not in args.environment else args.environment.replace("/","-")
    FEEDBACK_ID = "_".join(
        [args.algorithm, env_name, str(args.seed)]
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
    elif "metaworld" in args.environment:
        environment_name = args.environment.replace("metaworld-", "")
        environment = ppo_make_metaworld_env(environment_name, args.seed)
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
        env_name=args.environment,
    )

    train_reward_model(
        reward_model,
        MODEL_ID,
        args.feedback_type,
        dataset,
        maximum_epochs=100,
        split_ratio=0.85,
        environment=args.environment,
        cpu_count=cpu_count,
        num_ensemble_models=args.n_ensemble,
        enable_progress_bar=False if args.no_loading_bar else True,
        noise_level=args.noise_level,
        seed=args.seed,
    )



if __name__ == "__main__":
    main()
