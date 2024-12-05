import os
import pandas as pd
import gym
import torch
import random
import numpy as np
import argparse
import wandb
from pathlib import Path
from typing import Optional, Tuple, Any, Dict

from procgen import ProcgenGym3Env
from gym3.wrapper import Gym3ToGymnasium
from gym.wrappers import TransformObservation, FrameStackObservation
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.atari_wrappers import AtariWrapper
from rl_zoo3.utils import ppo_make_metaworld_env
from minigrid.wrappers import FlatObsWrapper

class TrainingUtils:
    @staticmethod
    def setup_environment(env_name: str, seed: Optional[int] = None) -> gym.Env:
        """Create and configure the environment based on the environment name."""
        if "procgen" in env_name:
            _, short_name, _ = env_name.split("-")
            environment = Gym3ToGymnasium(ProcgenGym3Env(num=1, env_name=short_name))
            environment = TransformObservation(environment, lambda obs: obs["rgb"], environment.observation_space)
        elif "ALE/" in env_name:
            environment = FrameStackObservation(AtariWrapper(gym.make(env_name)), 4)
            environment = TransformObservation(environment, lambda obs: obs.squeeze(-1), environment.observation_space)
        elif "MiniGrid" in env_name:
            environment = FlatObsWrapper(gym.make(env_name))
        elif "metaworld" in env_name:
            environment_name = env_name.replace("metaworld-", "")
            environment = ppo_make_metaworld_env(environment_name, seed) if seed else ppo_make_metaworld_env(environment_name)
        else:
            environment = gym.make(env_name)
        return environment

    @staticmethod
    def setup_base_parser() -> argparse.ArgumentParser:
        """Create a base argument parser with common arguments."""
        parser = argparse.ArgumentParser()
        parser.add_argument("--environment", type=str, default="HalfCheetah-v5", help="Environment name")
        parser.add_argument("--algorithm", type=str, default="ppo", help="RL algorithm")
        parser.add_argument("--seed", type=int, default=12, help="Random seed")
        parser.add_argument("--n-feedback", type=int, default=-1, help="Number of feedback instances")
        parser.add_argument("--noise-level", type=float, default=0.0, help="Noise level to add to feedback/demonstrations")
        return parser

    @staticmethod
    def set_seeds(seed: int) -> None:
        """Set random seeds for reproducibility."""
        np.random.seed(seed)
        random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)

    @staticmethod
    def get_device() -> str:
        """Get the appropriate device (CPU/CUDA)."""
        return "cuda" if torch.cuda.is_available() else "cpu"

    @staticmethod
    def get_model_ids(args: argparse.Namespace) -> Tuple[str, str]:
        """Generate feedback and model IDs based on arguments."""
        env_name = args.environment if "ALE" not in args.environment else args.environment.replace("/","-")
        feedback_id = f"{args.algorithm}_{env_name}_{args.seed}"
        model_id = f"{feedback_id}_{getattr(args, 'feedback_type', 'default')}_{args.seed}"
        
        if args.noise_level > 0.0:
            model_id = f"{model_id}_noise_{str(args.noise_level)}"
        if args.n_feedback != -1:
            model_id = f"{model_id}_nfeedback_{str(args.n_feedback)}"
            
        return feedback_id, model_id

    @staticmethod
    def load_expert_models(env_name: str, algorithm: str, checkpoints_path: str, 
                          environment: gym.Env, top_n_models: Optional[int] = None) -> list:
        """Load expert models for the given environment."""
        expert_model_paths = [
            os.path.join(checkpoints_path, algorithm, model) 
            for model in os.listdir(os.path.join(checkpoints_path, algorithm)) 
            if env_name in model
        ]

        if top_n_models:
            try:
                run_eval_scores = pd.read_csv(os.path.join(checkpoints_path, "collected_results.csv"))
                run_eval_scores = run_eval_scores.loc[
                    run_eval_scores['env'] == env_name
                ].sort_values(by=['eval_score'], ascending=False).head(top_n_models)["run"].to_list()
                expert_model_paths = [path for path in expert_model_paths if path.split(os.path.sep)[-1] in run_eval_scores]
            except:
                print("[WARN] No eval benchmark results available.")

        expert_models = []
        model_class = PPO if algorithm == "ppo" else SAC
        
        for expert_model_path in expert_model_paths:
            if os.path.isfile(os.path.join(expert_model_path, env_name, "vecnormalize.pkl")):
                norm_env = VecNormalize.load(
                    os.path.join(expert_model_path, env_name, "vecnormalize.pkl"),
                    DummyVecEnv([lambda: environment])
                )
            else:
                norm_env = None
                
            model_path = os.path.join(
                expert_model_path, 
                f"{env_name}.zip" if "ALE" not in env_name else "best_model.zip"
            )
            expert_models.append((model_class.load(model_path), norm_env))
            
        return expert_models

    @staticmethod
    def setup_wandb_logging(model_id: str, args: argparse.Namespace, 
                           additional_config: Optional[Dict[str, Any]] = None) -> Any:
        """Initialize W&B logging with given configuration."""
        config = {
            **vars(args),
            "seed": args.seed,
            "environment": args.environment,
            "n_feedback": args.n_feedback,
        }
        
        if additional_config:
            config.update(additional_config)
            
        return wandb.init(
            name=model_id,
            project="multi_reward_feedback",
            config=config,
            sync_tensorboard=True,
            monitor_gym=False,
            save_code=False,
        )