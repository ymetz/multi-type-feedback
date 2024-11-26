"""Module for training a behavioral cloning agent using demonstrations."""

import argparse
import os
import pickle
from os import path
from pathlib import Path
import numpy as np
import torch
from imitation.algorithms import bc
from imitation.data.types import Trajectory
import gymnasium as gym
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.utils import set_random_seed
import wandb
from imitation.data import rollout
from rl_zoo3.utils import ppo_make_metaworld_env

# register custom envs
import ale_py
import minigrid
import highway_env

def load_demonstrations(demo_path: str, noise_level: float = 0.0, discrete_action_space: bool = False):
    """Load and process demonstration data from pickle file."""
    with open(demo_path, "rb") as f:
        feedback_data = pickle.load(f)
    
    observations = []
    actions = []
    terms = []
    
    for demo in feedback_data["demos"]:
        obs = np.vstack([p[0] for p in demo])
        acts = np.vstack([p[1] for p in demo])
        dones = np.vstack([p[-1] for p in demo])
        
        if noise_level > 0.0:
            # Add noise to demonstrations if specified
            obs_min, obs_max = np.min(obs, axis=0), np.max(obs, axis=0)
            obs_diff = obs_max - obs_min
            acts_min, acts_max = np.min(acts, axis=0), np.max(acts, axis=0)
            acts_diff = acts_max - acts_min
            
            # Helper function for truncated Gaussian noise
            def truncated_gaussian_vectorized(mean, width, low, upp):
                samples = np.random.normal(loc=mean, scale=width)
                return np.clip(samples, low, upp)
            
            obs = truncated_gaussian_vectorized(
                mean=obs,
                width=np.array(noise_level) * obs_diff,
                low=obs_min,
                upp=obs_max
            )
            
            acts = truncated_gaussian_vectorized(
                mean=acts,
                width=np.array(noise_level) * acts_diff,
                low=acts_min,
                upp=acts_max
            )

        if len(acts) <= 1:
            continue # very short trajectory does not work
            print("SKIPPING VERY SHORT TRAJECTORY")
        
        observations.append(obs)

        # acts
        if discrete_action_space:
            acts = np.argmax(acts, axis=1)
        
        actions.append(acts[:-1])
        terms.append(dones)

    return [Trajectory(obs=flat_obs, acts=flat_acts, terminal=terms[-1], infos=[{} for _ in range(len(flat_acts))]) for (flat_obs, flat_acts, terms) in zip(observations, actions, terms)]

def main():
    """Run behavioral cloning training."""
    script_path = Path(__file__).parents[1].resolve()
    
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--environment",
        type=str,
        default="HalfCheetah-v5",
        help="Environment name",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=12,
        help="Random seed",
    )
    parser.add_argument(
        "--noise-level",
        type=float,
        default=0.0,
        help="Noise level to add to demonstrations",
    )
    parser.add_argument(
        "--n-epochs",
        type=int,
        default=20,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--algo",
        type=str,
        default="ppo",
        help="Number of training epochs",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for training",
    )
    args = parser.parse_args()

    FEEDBACK_ID = f"{args.algo}_{args.environment}_{args.seed}"
    if args.noise_level > 0.0:
        MODEL_ID = f"{FEEDBACK_ID}_demonstrative_{args.seed}_noise_{str(args.noise_level)}"
    else:
        MODEL_ID = f"{FEEDBACK_ID}_demonstrative_{args.seed}"

    # Set up wandb logging
    run = wandb.init(
        name="BC_"+MODEL_ID,
        project="multi_reward_feedback_final_lul",
        config={
            **vars(args),
            "model_type": "behavioral_cloning",
            "noise_level": args.noise_level,
            "seed": args.seed,
            "environment": args.environment,
        },
    )

    # Set random seed
    set_random_seed(args.seed)

    rng = np.random.default_rng(args.seed)
    
    if "procgen" in args.environment:
        _, short_name, _ = args.environment.split("-")
        environment = Gym3ToGymnasium(ProcgenGym3Env(num=1, env_name=short_name))
        environment = TransformObservation(environment, lambda obs: obs["rgb"], environment.observation_space)

        eval_env = Gym3ToGymnasium(ProcgenGym3Env(num=1, env_name=short_name))
        eval_env = TransformObservation(eval_env, lambda obs: obs["rgb"], eval_env.observation_space)
    elif "ALE/" in args.environment:
        environment = FrameStackObservation(AtariWrapper(gym.make(args.environment)), 4)
        environment = TransformObservation(environment, lambda obs: obs.squeeze(-1), environment.observation_space)

        eval_env = FrameStackObservation(AtariWrapper(gym.make(args.environment)), 4)
        eval_env = TransformObservation(eval_env, lambda obs: obs.squeeze(-1), eval_env.observation_space)
    elif "MiniGrid" in args.environment:
        environment = FlatObsWrapper(gym.make(args.environment))
        eval_env = FlatObsWrapper(gym.make(args.environment))
    elif "metaworld" in args.environment:
        environment_name = args.environment.replace("metaworld-", "")
        environment = ppo_make_metaworld_env(environment_name, args.seed)
        eval_env = ppo_make_metaworld_env(environment_name, args.seed)
    else:
        environment = gym.make(args.environment)
        eval_env = gym.make(args.environment)
    
    # Load demonstrations
    demo_path = os.path.join(script_path, "feedback_regen", f"{FEEDBACK_ID}.pkl")

    is_discrete_action = isinstance(environment.action_space, gym.spaces.Discrete)
    
    trajectories = load_demonstrations(demo_path, args.noise_level, discrete_action_space=is_discrete_action)

    trajectories = rollout.flatten_trajectories(trajectories)
    
    # Create BC trainer
    bc_trainer = bc.BC(
        observation_space=environment.observation_space,
        action_space=environment.action_space,
        demonstrations=trajectories,  # We'll manually pass transitions
        batch_size=args.batch_size,
        rng=rng,
        device="cuda:0",
        #learning_rate=args.learning_rate,
    )
    
    # Train the BC policy
    for epoch in range(args.n_epochs):
        stats = bc_trainer.train(
            n_epochs=1,
            progress_bar=False
        )
        print(stats)
        
        # Evaluate policy
        mean_reward, std_reward = evaluate_policy(
            bc_trainer.policy,
            eval_env,
            n_eval_episodes=10
        )
        print(f"Epoch {epoch}: Mean reward = {mean_reward:.2f} +/- {std_reward:.2f}")
        wandb.log({
            "epoch": epoch,
            "mean_reward": mean_reward,
            "std_reward": std_reward
        })
    
    # Save the trained policy
    save_path = os.path.join("agents", f"BC_{MODEL_ID}")
    os.makedirs(save_path, exist_ok=True)
    bc_trainer.policy.save(os.path.join(save_path, "bc_policy"))
    
    # Final evaluation
    mean_reward, std_reward = evaluate_policy(
        bc_trainer.policy,
        eval_env,
        n_eval_episodes=100
    )
    print(f"Final evaluation: Mean reward = {mean_reward:.2f} +/- {std_reward:.2f}")
    wandb.log({
        "final_mean_reward": mean_reward,
        "final_std_reward": std_reward,
    })

if __name__ == "__main__":
    main()