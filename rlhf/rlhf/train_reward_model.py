"""Module for training a reward model from the generated feedback."""

import argparse
import math
import os
import pickle
import random
from os import path
from pathlib import Path
from random import randint, randrange
from typing import List, Tuple, Union

import ale_py
import gymnasium as gym
import highway_env
import minigrid
import numpy as np
import torch
import wandb
from gymnasium.wrappers.stateful_observation import FrameStackObservation
from gymnasium.wrappers.transform_observation import TransformObservation
from minigrid.wrappers import FlatObsWrapper
from numpy.typing import NDArray
from procgen import ProcgenGym3Env
from pytorch_lightning import Callback, LightningModule, Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import WandbLogger
from rl_zoo3.utils import ppo_make_metaworld_env
from rl_zoo3.wrappers import Gym3ToGymnasium
from scipy.stats import truncnorm, uniform
from stable_baselines3.common.atari_wrappers import AtariWrapper
from stable_baselines3.common.vec_env import VecExtractDictObs
from torch.utils.data import DataLoader, Dataset, random_split

from rlhf.datatypes import FeedbackDataset, FeedbackType, SegmentT
from rlhf.networks import (
    LightningCnnNetwork,
    LightningNetwork,
    calculate_pairwise_loss,
    calculate_single_reward_loss,
)
from rlhf.utils import TrainingUtils

# for convenice sake, todo: make dynamic in the future
discount_factors = {
    "HalfCheetah-v5": 0.98,
    "Hopper-v5": 0.99,
    "Swimmer-v5": 0.9999,
    "Ant-v5": 0.99,
    "Walker2d-v5": 0.99,
    "ALE/BeamRider-v5": 0.99,
    "ALE/MsPacman-v5": 0.99,
    "ALE/Enduro-v5": 0.99,
    "ALE/Pong-v5": 0.99,
    "Humanoid-v5": 0.99,
    "highway-fast-v0": 0.8,
    "merge-v0": 0.8,
    "roundabout-v0": 0.8,
    "metaworld-sweep-into-v2": 0.99,
    "metaworld-button-press-v2": 0.99,
    "metaworld-pick-place-v2": 0.99,
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
    lower = mean - width / 2
    upper = mean + width / 2

    # Clip the bounds to [low, upp]
    lower = np.clip(lower, low, upp)
    upper = np.clip(upper, low, upp)

    # Generate random values
    r = np.random.uniform(size=mean.shape)

    # Calculate the result
    result = lower + r * (upper - lower)

    return result[0] if scalar_input else result


def truncated_gaussian_vectorized(mean, width, low=0, upp=9, min_width=1e-6):
    """
    Generate samples from a truncated Gaussian distribution with proper handling of zero variance.

    Args:
        mean: Mean of the distribution
        width: Width parameter (typically proportional to standard deviation)
        low: Lower bound of the truncation
        upp: Upper bound of the truncation
        min_width: Minimum allowed width to prevent numerical issues
    """
    # Handle scalar inputs
    scalar_input = np.isscalar(mean) and np.isscalar(width)
    mean = np.atleast_1d(mean)
    width = np.atleast_1d(width)

    # Ensure width is at least min_width to prevent division by zero
    width = np.maximum(width, min_width)

    # Calculate the bounds of the distribution
    lower = np.maximum(mean - width / 2, low)
    upper = np.minimum(mean + width / 2, upp)

    # Calculate parameters for truncated normal distribution
    sigma = width / 4  # 4 sigma range
    a = (lower - mean) / sigma
    b = (upper - mean) / sigma

    result = np.where(
        width <= min_width * 1.1,  # Using 1.1 to account for floating point comparison
        mean + np.random.normal(0, min_width / 10, size=mean.shape),  # Tiny noise
        truncnorm.rvs(a, b, loc=mean, scale=sigma, size=mean.shape),
    )

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
        noise_level: float = 0.0,  # depending on the feedback, we use different types of noise (e.g. flip the preference, add noise to the rating or description)
        segment_len: int = 50,
        env=None,
        seed: int = 1234,
    ):
        """Initialize dataset."""
        print("Loading dataset...")

        self.targets: Union[
            List[SegmentT],
            List[NDArray],
            Tuple[SegmentT, SegmentT],
            Tuple[NDArray, NDArray],
        ] = []
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
                    actions = torch.cat(
                        [actions, torch.zeros(pad_size, *actions.shape[1:])], dim=0
                    )

                self.targets.append((obs, actions))

            self.preds = feedback_data["ratings"]
            # add noise to the ratings
            if noise_level > 0:
                # apply noise, accounting for clipping at the borders [0,9]
                self.preds = truncated_gaussian_vectorized(
                    mean=self.preds,
                    width=noise_level * 10 * np.ones_like(self.preds),
                    low=0,
                    upp=9,
                )
        elif feedback_type == "comparative":

            rews_min, rews_max = np.min(
                [e * -1 for e in feedback_data["opt_gaps"]]
            ), np.max([e * -1 for e in feedback_data["opt_gaps"]])
            ref_diff = np.abs(rews_max - rews_min)

            flipped = 0
            for comp in feedback_data["preferences"]:
                # seg 1
                obs = torch.vstack(
                    [
                        torch.as_tensor(p[0]).float()
                        for p in feedback_data["segments"][comp[0]]
                    ]
                )
                actions = torch.vstack(
                    [
                        torch.as_tensor(p[1]).float()
                        for p in feedback_data["segments"][comp[0]]
                    ]
                )

                # seg 2
                obs2 = torch.vstack(
                    [
                        torch.as_tensor(p[0]).float()
                        for p in feedback_data["segments"][comp[1]]
                    ]
                )
                actions2 = torch.vstack(
                    [
                        torch.as_tensor(p[1]).float()
                        for p in feedback_data["segments"][comp[1]]
                    ]
                )

                # Pad both trajectories to the maximum length, necessary for batching with data loader
                len_obs = obs.size(0)
                len_obs2 = obs2.size(0)

                if len_obs < segment_len:
                    pad_size = segment_len - len_obs
                    obs = torch.cat([obs, torch.zeros(pad_size, *obs.shape[1:])], dim=0)
                    actions = torch.cat(
                        [actions, torch.zeros(pad_size, *actions.shape[1:])], dim=0
                    )
                if len_obs2 < segment_len:
                    pad_size = segment_len - len_obs2
                    obs2 = torch.cat(
                        [obs2, torch.zeros(pad_size, *obs2.shape[1:])], dim=0
                    )
                    actions2 = torch.cat(
                        [actions2, torch.zeros(pad_size, *actions2.shape[1:])], dim=0
                    )

                # add noise and recompute preferences
                if noise_level > 0:
                    rew1 = -feedback_data["opt_gaps"][comp[0]]
                    rew2 = -feedback_data["opt_gaps"][comp[1]]

                    rew1 = truncated_gaussian_vectorized(
                        mean=np.array(rew1),
                        width=np.array(noise_level) * ref_diff,
                        low=rews_min,
                        upp=rews_max,
                    )
                    rew2 = truncated_gaussian_vectorized(
                        mean=np.array(rew2),
                        width=np.array(noise_level) * ref_diff,
                        low=rews_min,
                        upp=rews_max,
                    )

                    if rew2 > rew1:
                        self.targets.append(((obs, actions), (obs2, actions2)))
                    else:
                        self.targets.append(((obs2, actions2), (obs, actions)))
                        flipped += 1
                    self.preds.append(comp[2])
                else:
                    self.targets.append(((obs, actions), (obs2, actions2)))
                    self.preds.append(comp[2])

        elif feedback_type == "demonstrative":

            with open(
                os.path.join("samples", f"random_{env_name}.pkl"), "rb"
            ) as random_file:
                random_data = pickle.load(random_file)

            for demo in feedback_data["demos"]:
                obs = np.vstack([p[0] for p in demo])
                actions = np.vstack([p[1] for p in demo])

                if noise_level > 0.0:

                    # Calculate statistics across all data points, keeping the feature dimensions
                    obs_min = np.min(obs, axis=0)
                    obs_max = np.max(obs, axis=0)
                    obs_std = np.std(obs, axis=0)

                    acts_min = np.min(actions, axis=0)
                    acts_max = np.max(actions, axis=0)
                    acts_std = np.std(actions, axis=0)

                    # Process each batch separately
                    noisy_obs = []
                    noisy_actions = []

                    for i in range(obs.shape[0]):
                        # Add noise to each batch independently
                        noisy_obs.append(
                            truncated_gaussian_vectorized(
                                mean=obs[i],
                                width=np.array(noise_level)
                                * obs_std,  # obs_std is already per-feature
                                low=obs_min,
                                upp=obs_max,
                            )
                        )

                        noisy_actions.append(
                            truncated_gaussian_vectorized(
                                mean=actions[i],
                                width=np.array(noise_level) * acts_std,
                                low=acts_min,
                                upp=acts_max,
                            )
                        )

                    obs = np.stack(noisy_obs)
                    actions = np.stack(noisy_actions)

                obs = torch.as_tensor(obs).float()
                actions = torch.as_tensor(actions).float()

                # just use a random segment as the opposite
                rand_index = random.randrange(0, len(random_data["segments"]))
                obs_rand = torch.vstack(
                    [
                        torch.as_tensor(p[0]).float()
                        for p in random_data["segments"][rand_index]
                    ]
                )
                actions_rand = torch.vstack(
                    [
                        torch.as_tensor(p[1]).float()
                        for p in random_data["segments"][rand_index]
                    ]
                )

                # Pad both trajectories to the maximum length
                len_obs = obs.size(0)
                len_obs_rand = obs_rand.size(0)

                if len_obs < segment_len:
                    pad_size = segment_len - len_obs
                    obs = torch.cat([obs, torch.zeros(pad_size, *obs.shape[1:])], dim=0)
                    actions = torch.cat(
                        [actions, torch.zeros(pad_size, *actions.shape[1:])], dim=0
                    )
                if len_obs_rand < segment_len:
                    pad_size = segment_len - len_obs_rand
                    obs_rand = torch.cat(
                        [obs_rand, torch.zeros(pad_size, *obs_rand.shape[1:])], dim=0
                    )
                    actions_rand = torch.cat(
                        [actions_rand, torch.zeros(pad_size, *actions_rand.shape[1:])],
                        dim=0,
                    )

                self.targets.append(((obs_rand, actions_rand), (obs, actions)))
                self.preds.append(
                    1
                )  # assume that the demonstration is optimal, maybe add confidence value (based on regret)
        elif feedback_type == "corrective":

            rews_min, rews_max = np.min(
                [e * -1 for e in feedback_data["opt_gaps"]]
            ), np.max([e * -1 for e in feedback_data["opt_gaps"]])
            rew_diff = np.abs(rews_max - rews_min)
            gamma = discount_factors[env_name]

            flipped = 0
            for comp in feedback_data["corrections"]:
                obs = torch.vstack([torch.as_tensor(p[0]).float() for p in comp[0]])
                actions = torch.vstack([torch.as_tensor(p[1]).float() for p in comp[0]])

                obs2 = torch.vstack([torch.as_tensor(p[0]).float() for p in comp[1]])
                actions2 = torch.vstack(
                    [torch.as_tensor(p[1]).float() for p in comp[1]]
                )

                # Pad both trajectories to the maximum length
                len_obs = obs.size(0)
                len_obs2 = obs2.size(0)

                if len_obs < segment_len:
                    pad_size = segment_len - len_obs
                    obs = torch.cat([obs, torch.zeros(pad_size, *obs.shape[1:])], dim=0)
                    actions = torch.cat(
                        [actions, torch.zeros(pad_size, *actions.shape[1:])], dim=0
                    )
                if len_obs2 < segment_len:
                    pad_size = segment_len - len_obs2
                    obs2 = torch.cat(
                        [obs2, torch.zeros(pad_size, *obs2.shape[1:])], dim=0
                    )
                    actions2 = torch.cat(
                        [actions2, torch.zeros(pad_size, *actions2.shape[1:])], dim=0
                    )

                # add noise and recompute preferences
                if noise_level > 0.0:
                    rews1 = discounted_sum_numpy(
                        np.array([p[2] for p in comp[0]]), gamma
                    )
                    rews2 = discounted_sum_numpy(
                        np.array([p[2] for p in comp[1]]), gamma
                    )

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
                        upp=max(rews_max, rews2),
                    ).item()

                    if rew2 > rew1:
                        self.targets.append(((obs, actions), (obs2, actions2)))
                    else:
                        self.targets.append(((obs2, actions2), (obs, actions)))
                        flipped += 1
                    self.preds.append(1)
                else:
                    self.targets.append(((obs, actions), (obs2, actions2)))
                    self.preds.append(1)
        elif feedback_type == "descriptive":

            cluster_rews = np.array([cr[2] for cr in feedback_data["description"]])
            cluster_rew_min, cluster_rew_max = cluster_rews.min(), cluster_rews.max()
            cluster_rew_diff = np.abs(cluster_rew_max - cluster_rew_min)

            for cluster_representative in feedback_data["description"]:
                self.targets.append(
                    (
                        torch.as_tensor(cluster_representative[0]).unsqueeze(0).float(),
                        torch.as_tensor(cluster_representative[1]).unsqueeze(0).float(),
                    )
                )

                if noise_level > 0.0:
                    rew = truncated_gaussian_vectorized(
                        mean=np.array(cluster_representative[2]),
                        width=np.array(noise_level) * cluster_rew_diff,
                        low=cluster_rew_min,
                        upp=cluster_rew_max,
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
                obs = (
                    torch.as_tensor(feedback_data["description"][idx_1][0])
                    .unsqueeze(0)
                    .float()
                )
                actions = (
                    torch.as_tensor(feedback_data["description"][idx_1][1])
                    .unsqueeze(0)
                    .float()
                )

                idx_2 = cpref[1]

                # cluster 2
                obs2 = (
                    torch.as_tensor(feedback_data["description"][idx_2][0])
                    .unsqueeze(0)
                    .float()
                )
                actions2 = (
                    torch.as_tensor(feedback_data["description"][idx_2][1])
                    .unsqueeze(0)
                    .float()
                )

                # add noise and recompute preferences
                if noise_level > 0:
                    rew1 = feedback_data["description"][idx_1][2]
                    rew2 = feedback_data["description"][idx_2][2]

                    rew1 = truncated_gaussian_vectorized(
                        mean=np.array(rew1),
                        width=np.array(noise_level) * cluster_rew_diff,
                        low=cluster_rew_min,
                        upp=cluster_rew_max,
                    ).item()
                    rew2 = truncated_gaussian_vectorized(
                        mean=np.array(rew2),
                        width=np.array(noise_level) * cluster_rew_diff,
                        low=cluster_rew_min,
                        upp=cluster_rew_max,
                    ).item()

                    if rew2 > rew1:
                        self.targets.append(((obs, actions), (obs2, actions2)))
                    else:
                        self.targets.append(((obs2, actions2), (obs, actions)))
                        flipped += 1
                    self.preds.append(cpref[2])
                else:
                    self.targets.append(((obs, actions), (obs2, actions2)))
                    self.preds.append(cpref[2])
        else:
            raise NotImplementedError("Dataset not implemented for this feedback type.")

        print("Dataset loaded")
        print(f"N TARGETS AVAILABLE: {len(self.targets)}, N_FEEDBACK: {n_feedback}")

        if n_feedback != -1 and n_feedback < len(self.targets):
            # is a bit inefficient as we first collected the entire dataset..but we just have to do it once
            rng = np.random.default_rng(seed)
            indices = rng.choice(len(self.targets), size=n_feedback, replace=False)

            self.targets = [self.targets[i] for i in indices]
            self.preds = [self.preds[i] for i in indices]

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
    n_feedback: int = -1,
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
        # num_workers=cpu_count,
        num_workers=0,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_set,
        batch_size=num_ensemble_models,
        pin_memory=True,
        num_workers=1,
        drop_last=True,
    )

    checkpoint_callback = ModelCheckpoint(
        dirpath=path.join(script_path, "reward_models_lul"),
        filename=reward_model_id,
        monitor="val_loss",
    )

    # initialise the wandb logger and name your wandb project
    wandb_logger = WandbLogger(
        project="multi_reward_feedback_final_lul",
        name=reward_model_id,
        config={
            "feedback_type": feedback_type,
            "noise_level": noise_level,
            "seed": seed,
            "environment": environment,
            "n_feedback": n_feedback,
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
    parser = TrainingUtils.setup_base_parser()
    parser.add_argument(
        "--feedback-type", type=str, default="evaluative", help="Type of feedback"
    )
    parser.add_argument(
        "--n-ensemble", type=int, default=4, help="Number of ensemble models"
    )
    parser.add_argument(
        "--no-loading-bar", action="store_true", help="Disable loading bar"
    )
    args = parser.parse_args()

    TrainingUtils.set_seeds(args.seed)
    environment = TrainingUtils.setup_environment(args.environment)

    feedback_id, model_id = TrainingUtils.get_model_ids(args)

    # Setup reward model
    reward_model = (
        LightningCnnNetwork
        if "procgen" in args.environment or "ALE" in args.environment
        else LightningNetwork
    )(
        input_spaces=(environment.observation_space, environment.action_space),
        hidden_dim=256,
        action_hidden_dim=(
            16 if "procgen" in args.environment or "ALE" in args.environment else 32
        ),
        layer_num=(
            3 if "procgen" in args.environment or "ALE" in args.environment else 6
        ),
        cnn_channels=(
            (16, 32, 32)
            if "procgen" in args.environment or "ALE" in args.environment
            else None
        ),
        output_dim=1,
        loss_function=(
            calculate_single_reward_loss
            if args.feedback_type in ["evaluative", "descriptive"]
            else calculate_pairwise_loss
        ),
        learning_rate=1e-5,
        ensemble_count=args.n_ensemble,
    )

    dataset = FeedbackDataset(
        os.path.join(script_path, "feedback_regen", f"{feedback_id}.pkl"),
        args.feedback_type,
        args.n_feedback,
        noise_level=args.noise_level,
        env=environment if args.feedback_type == "demonstrative" else None,
        env_name=args.environment,
        seed=args.seed,
    )

    train_reward_model(
        reward_model,
        model_id,
        args.feedback_type,
        dataset,
        maximum_epochs=100,
        split_ratio=0.85,
        environment=args.environment,
        cpu_count=os.cpu_count() or 8,
        num_ensemble_models=args.n_ensemble,
        enable_progress_bar=not args.no_loading_bar,
        noise_level=args.noise_level,
        n_feedback=args.n_feedback,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
