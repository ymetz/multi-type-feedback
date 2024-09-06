"""Module for training an RL agent."""

import argparse
import os
import sys
import typing
from os import path
from pathlib import Path
import wandb
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

from rlhf.common import get_reward_model_name
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

    def __call__(
        self,
        state: numpy.ndarray,
        actions: numpy.ndarray,
        next_state: numpy.ndarray,
        _done: numpy.ndarray,
    ) -> list:
        """Return reward given the current state."""
        with torch.no_grad():
            return self.reward_model(
                    torch.as_tensor(state, device=self.device, dtype=torch.float),
                    torch.as_tensor(actions, device=self.device, dtype=torch.float)
            ).squeeze(1).cpu().numpy()

def main():
    """Run RL agent training."""

    script_path = Path(__file__).parents[1].resolve()

    cpu_count = os.cpu_count()
    cpu_count = cpu_count if cpu_count is not None else 8

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--experiment", 
        type=int, 
        default=0, 
        help="Experiment number",
    )
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
        "--train-steps", 
        type=int, 
        default=int(1e6), 
        help="Number of steps to generate feedback for",
    )
    parser.add_argument(
        "--seed", 
        type=int, 
        default=1337, 
        help="TODO: Seed for env and stuff",
    )

    args = parser.parse_args()

    FEEDBACK_ID = "_".join(
        [args.algorithm, args.environment]
    )
    MODEL_ID = f"{FEEDBACK_ID}_{args.feedback_type}_{args.experiment}"

    reward_model_path = os.path.join(script_path, "reward_models", MODEL_ID + ".ckpt")

    print("Reward model ID:", MODEL_ID)

    set_random_seed(args.seed)

    run = wandb.init(
        name="RL_"+MODEL_ID,
        project="multi_reward_feedback",
        config=vars(args),
        sync_tensorboard=True,  # auto-upload sb3's tensorboard metrics
        monitor_gym=False,  # auto-upload the videos of agents playing the game
        save_code=False,  # optional
    )


    # ================ Load correct reward function model =================
    match args.environment:
        case "Ant-v5" | "Humanoid-v5" | "HalfCheetah-v5":
            architecture_cls = LightningNetwork
            
        case "MiniGrid-GoToDoor-5x5-v0" | "procgen-coinrun-v0" | "procgen-miner-v0" | "procgen-maze-v0" | "ALE/MsPacman-v5" | "ALE/Pong-v5":
            architecture_cls = LightningCnnNetwork

    # ================ Load correct reward function model ===================

    exp_manager = ExperimentManager(
        args,
        args.algorithm,
        args.environment,
        os.path.join("agents","RL_"+MODEL_ID),
        tensorboard_log=f"runs/{"RL_"+MODEL_ID}",
        n_timesteps=args.train_steps,
        seed=args.seed,
        log_interval=-1,
        reward_function=CustomReward(
            reward_model_cls=architecture_cls,
            reward_model_path=reward_model_path,
            device=DEVICE
        ),
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
