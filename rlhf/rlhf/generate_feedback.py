import argparse
import os
import pickle
import re
from pathlib import Path
from typing import Type, Union

import gymnasium as gym
import numpy as np
import torch
from captum.attr import IntegratedGradients
from numpy.typing import NDArray
from stable_baselines3 import PPO, SAC
from torch import Tensor

from rlhf.datatypes import ActionNumpyT, Feedback, ObservationT

def get_attributions(
    observation: Tensor,
    actions: NDArray[ActionNumpyT],
    explainer: IntegratedGradients,
    algorithm: str = "sac",
    device: str = "cuda",
) -> NDArray[np.float64]:
    """
    Compute attributions for a given observation and actions using the provided explainer.
    """
    obs_baselines = torch.zeros_like(observation)
    if algorithm == "sac":
        actions_tensor = torch.from_numpy(actions).unsqueeze(0).to(device)
        actions_baselines = torch.zeros_like(actions_tensor)
    
        attributions = explainer.attribute(
            (observation, actions_tensor),
            target=0,
            baselines=(obs_baselines, actions_baselines),
            internal_batch_size=64,
        )
    else:
        attributions = explainer.attribute(
            observation,
            target=0,
            baselines=obs_baselines,
            internal_batch_size=64,
        )        

    return attributions.squeeze().cpu().numpy()

def predict_expert_value(
    expert_model: Union[PPO, SAC], 
    observation: Tensor, 
    actions: Tensor
) -> Tensor:
    """Return the value from the expert's value function for a given observation and actions."""
    with torch.no_grad():
        return torch.min(
            torch.cat(expert_model.policy.critic_target(observation, actions), dim=1) if isinstance(expert_model, SAC) else expert_model.policy.predict_values(observation),
            dim=1,
            keepdim=True,
        )[0]

def get_model_logits(
    expert_model: Union[PPO, SAC],
    observation: Tensor,
    actions: Tensor = None,
) -> Tensor:
    if isinstance(expert_model, SAC):
        return torch.min(
            torch.cat(expert_model.policy.critic_target(observation, actions), dim=1) if isinstance(expert_model, SAC) else expert_model.policy.predict_values(observation),
            dim=1,
            keepdim=True,
        )[0]
    else:
        return expert_model.policy.predict_values(observation)

def generate_feedback(
    model_class: Type[Union[PPO, SAC]],
    expert_model: Union[PPO, SAC],
    environment: gym.Env,
    environment_name: str = "HalfCheetah-v3",
    checkpoints_path: str = "rl_checkpoints",
    steps_per_checkpoint: int = 1000,
    algorithm: str = "sac",
    device: str = "cuda",
) -> list[Feedback[ObservationT, ActionNumpyT]]:
    """Generate agent's observations and feedback in the training environment."""
    feedback = []
    feedback_id = f"{algorithm}_{environment_name}"
    checkpoints_dir = os.path.join(checkpoints_path, algorithm, f"{environment_name}_1")

    print(f"Generating feedback for: {feedback_id}")

    checkpoint_files = [
        file for file in os.listdir(checkpoints_dir) if re.search(r"rl_model_.*\.zip", file)
    ] or [f"{environment_name}.zip"]

    explainer_cls = IntegratedGradients

    for model_file in checkpoint_files:
        model = model_class.load(
            os.path.join(checkpoints_dir, model_file),
            custom_objects={"learning_rate": 0.0, "lr_schedule": lambda _: 0.0},
        )

        if algorithm == "sac":
            explainer = explainer_cls(lambda obs, acts: get_model_logits(expert_model, obs, acts))
        else:
            explainer = explainer_cls(lambda obs: get_model_logits(expert_model, obs))

        prev_expert_value = 0
        observation, _ = environment.reset()

        for _ in range(steps_per_checkpoint):
            # state_copy = environment.sim.get_state()  # type: ignore

            expert_observation = observation
            for _ in range(20):
                expert_actions, _ = expert_model.predict(expert_observation, deterministic=True)
                next_expert_observation, reward, terminated, _, _ = environment.step(expert_actions)
                expert_observation = next_expert_observation if not terminated else environment.reset()[0]

            # environment.sim.set_state(state_copy)  # type: ignore

            actions, _ = model.predict(observation, deterministic=True)
            next_observation, reward, terminated, _, _ = environment.step(actions)

            observation_tensor = expert_model.policy.obs_to_tensor(np.array(observation))[0]
            expert_value = predict_expert_value(
                expert_model, observation_tensor, torch.from_numpy(actions).unsqueeze(0).to(device)
            ).item()

            expert_own_value = predict_expert_value(
                expert_model,
                expert_model.policy.obs_to_tensor(np.array(expert_observation))[0],
                torch.from_numpy(expert_actions).unsqueeze(0).to(device)
            ).item()

            feedback.append(
                {
                    "actions": actions,
                    "observation": observation,
                    "next_observation": next_observation,
                    "reward": float(reward),
                    "expert_value": expert_value,
                    "expert_value_difference": expert_value - prev_expert_value,
                    "expert_observation": expert_observation,
                    "expert_actions": expert_actions,
                    "expert_value_attributions": get_attributions(
                        observation_tensor, actions, explainer, device=device, algorithm=algorithm
                    ),
                    "expert_own_value": expert_own_value,
                }
            )

            prev_expert_value = expert_value
            observation = next_observation if not terminated else environment.reset()[0]

    return feedback[1:]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", type=int, default=0, help="Experiment number")
    parser.add_argument("--feedback_type", type=str, default="evaluative", help="Type of feedback")
    parser.add_argument("--algorithm", type=str, default="sac", help="RL algorithm")
    parser.add_argument("--environment", type=str, default="HalfCheetah-v3", help="Environment")
    parser.add_argument("--use-sde", action="store_true", help="Use SDE in the RL algorithm")
    parser.add_argument("--use-reward-difference", action="store_true", help="Use reward difference")
    parser.add_argument("--steps-per-checkpoint", type=int, default=1000, help="Steps per checkpoint")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    feedback_id = f"{args.algorithm}_{args.environment}{'_sde' if args.use_sde else ''}"
    feedback_path = Path(__file__).parent.resolve() / "feedback" / f"{feedback_id}.pkl"
    checkpoints_path = "../main/logs"

    expert_model = (PPO if args.algorithm == "ppo" else SAC).load(
        os.path.join(checkpoints_path, args.algorithm, f"{args.environment}_1", f"{args.environment}.zip"),
        custom_objects={"learning_rate": 0.0, "lr_schedule": lambda _: 0.0, "clip_range": lambda _: 0.0},
    )

    environment = gym.make(args.environment)
    model_class = PPO if args.algorithm == "ppo" else SAC

    feedback = generate_feedback(
        model_class,
        expert_model,
        environment,
        environment_name=args.environment,
        steps_per_checkpoint=args.steps_per_checkpoint,
        checkpoints_path=checkpoints_path,
        algorithm=args.algorithm,
        device=device,
    )

    feedback_path.parent.mkdir(parents=True, exist_ok=True)
    print(feedback_path)
    with open(feedback_path, "wb") as feedback_file:
        pickle.dump(feedback, feedback_file, protocol=pickle.HIGHEST_PROTOCOL)

if __name__ == "__main__":
    main()
