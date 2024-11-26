"""Module for training an RL agent."""

import argparse
import os
import sys
import typing
from os import path
from pathlib import Path
import wandb
import numpy as np
from wandb.integration.sb3 import WandbCallback

import numpy
import gymnasium as gym
import pytorch_lightning as pl
import torch
from imitation.rewards.reward_function import RewardFn
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.utils import set_random_seed

# register custom envs
import ale_py
import minigrid
import highway_env

from rlhf.datatypes import FeedbackType
from rlhf.networks import LightningNetwork, LightningCnnNetwork, calculate_pairwise_loss, calculate_single_reward_loss

from rl_zoo3.exp_manager import ExperimentManager
from rl_zoo3.utils import ALGOS, StoreDict

class CustomReward(RewardFn):
    """Custom reward based on fine-tuned reward model."""

    def __init__(
        self,
        reward_model_cls: typing.Union[LightningNetwork, LightningCnnNetwork] = None,
        reward_model_path: list[str] = [],
        vec_env_norm_fn: typing.Callable = None,
        device: str = "cuda",
    ):
        """Initialize custom reward."""
        super().__init__()
        self.device = device

        self.reward_model = reward_model_cls.load_from_checkpoint(
            reward_model_path,
            map_location=device
        )

        self.rewards = []
        self.expert_rewards = []
        self.counter = 0
        self.n_discrete_actions = 5 # hard-code for highway-env for now

    def _one_hot_encode_batch(self, actions: torch.Tensor) -> torch.Tensor:
        """
        Convert nested batch of discrete actions to one-hot encoded format.
        
        Args:
            actions: Tensor of shape (1, batch_size) containing discrete action indices
            
        Returns:
            one_hot_actions: Tensor of shape (1, batch_size, n_discrete_actions)
        """
        outer_batch, inner_batch = actions.shape
        one_hot = torch.zeros((outer_batch, inner_batch, self.n_discrete_actions), device=self.device)
        actions = actions.long().unsqueeze(-1)  # Add dimension for scatter
        return one_hot.scatter_(2, actions, 1)
    
    def __call__(
        self,
        state: numpy.ndarray,
        actions: numpy.ndarray,
        next_state: numpy.ndarray,
        _done: numpy.ndarray,
    ) -> list:
        """Return reward given the current state."""
        
        state = torch.as_tensor(state, device=self.device, dtype=torch.float).unsqueeze(0)        
        actions = torch.as_tensor(actions, device=self.device, dtype=torch.float).unsqueeze(0)

        if len(actions.shape) < 3:
            actions = self._one_hot_encode_batch(actions)
        
        with torch.no_grad():
            if self.reward_model.ensemble_count > 1:
                state = state.expand(self.reward_model.ensemble_count, *state.shape[1:])
                actions = actions.expand(self.reward_model.ensemble_count, *actions.shape[1:])
            
            rewards = self.reward_model(
                torch.as_tensor(state, device=self.device, dtype=torch.float),
                torch.as_tensor(actions, device=self.device, dtype=torch.float)
            )
            # Reshape rewards to always have 3 dimensions: (ensemble_count, batch_size, 1)
            rewards = rewards.view(self.reward_model.ensemble_count, -1, 1)
            # Take mean across ensemble dimension (dim=0)
            mean_rewards = torch.mean(rewards, dim=0).squeeze(-1)
            
            return mean_rewards.cpu().numpy()

def main():
    """Run RL agent training."""

    script_path = Path(__file__).parents[1].resolve()

    cpu_count = os.cpu_count()
    cpu_count = cpu_count if cpu_count is not None else 8

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--algorithm", 
        type=str, 
        default="ppo", 
        help="RL algorithm",
    )
    parser.add_argument(
        "--feedback-type",
        type=str,
        default="evaluative",
        help="Type of feedback to train the reward model",
    )
    parser.add_argument(
        "--environment", 
        type=str, 
        default="HalfCheetah-v5", 
        help="Environment",
    )
    parser.add_argument(
        "--seed", 
        type=int, 
        default=12, 
        help="TODO: Seed for env and stuff",
    )
    parser.add_argument(
        "--noise-level",
        type=float,
        default=0.0,
        help="Noise level to add to the feedback",
    )
    args = parser.parse_args()

    FEEDBACK_ID = "_".join(
        [args.algorithm, args.environment, str(args.seed)]
    )
    if args.noise_level > 0.0:
        MODEL_ID = f"{FEEDBACK_ID}_{args.feedback_type}_{args.seed}_noise_{str(args.noise_level)}"
    else:
        MODEL_ID = f"{FEEDBACK_ID}_{args.feedback_type}_{args.seed}"

    if not args.feedback_type == "baseline":
        reward_model_path = os.path.join(script_path, "reward_models_lul", MODEL_ID + ".ckpt")

    print("Reward model ID:", MODEL_ID)

    set_random_seed(args.seed)

    run = wandb.init(
        name="RL_"+MODEL_ID,
        project="multi_reward_feedback_final_lul",
        config={
            **vars(args),
            "feedback_type": args.feedback_type,
            "noise_level": args.noise_level,
            "seed": args.seed,
            "environment": args.environment,
        },
        sync_tensorboard=True,  # auto-upload sb3's tensorboard metrics
        monitor_gym=False,  # auto-upload the videos of agents playing the game
        save_code=False,  # optional
    )


    # ================ Load correct reward function model =================
    if "ALE/" in args.environment or "procgen" in args.environment:
        architecture_cls = LightningCnnNetwork
    else:
        architecture_cls = LightningNetwork

    # ================ Load correct reward function model ===================

    exp_manager = ExperimentManager(
        args,
        args.algorithm,
        args.environment,
        os.path.join("agents","RL_"+MODEL_ID),
        tensorboard_log=f"runs/{'RL_'+MODEL_ID}",
        seed=args.seed,
        log_interval=-1,
        reward_function=CustomReward(
            reward_model_cls=architecture_cls,
            reward_model_path=reward_model_path,
            device=DEVICE,
        ) if not args.feedback_type == "baseline" else None,
        use_wandb_callback=True,
    )

    # Prepare experiment and launch hyperparameter optimization if needed
    results = exp_manager.setup_experiment()
    if results is not None:
        model, saved_hyperparams = results
        # we need to save the loaded hyperparameters
        args.saved_hyperparams = saved_hyperparams
        assert run is not None  # make mypy happy
        run.config.setdefaults(vars(args))

        # Normal training
        if model is not None:
            exp_manager.learn(model)
            exp_manager.save_trained_model(model)


if __name__ == "__main__":
    main()
