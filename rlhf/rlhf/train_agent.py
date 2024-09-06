"""Module for training an RL agent."""

import argparse
import os
import sys
import typing
from os import path
from pathlib import Path

import matplotlib
import numpy
import pytorch_lightning as pl
import torch
from imitation.rewards.reward_function import RewardFn
from imitation.rewards.reward_wrapper import RewardVecEnvWrapper
from imitation.util.util import make_vec_env
from stable_baselines3.ppo.ppo import PPO
from stable_baselines3.sac.sac import SAC

from rlhf.common import get_reward_model_name
from rlhf.datatypes import FeedbackType
from rlhf.networks import LightningNetwork

# Uncomment line below to use PyPlot with VSCode Tunnels
matplotlib.use("agg")

random_generator = numpy.random.default_rng(0)


class CustomReward(RewardFn):
    """Custom reward based on fine-tuned reward model."""

    def __init__(
        self,
        is_expert_reward: bool = False,
        reward_model_paths: list[str] = [],
        device: str = "cuda",
        expert_model: typing.Optional[SAC | PPO] = None,
    ):
        """Initialize custom reward."""
        super().__init__()

        self.reward_models: list[LightningNetwork] = []
        self.is_expert_reward = is_expert_reward
        self.reward_model_paths = reward_model_paths
        self.device = device
        self.expert_model = expert_model

        if not is_expert_reward:
            for reward_model_path in reward_model_paths:
                # pylint: disable=no-value-for-parameter
                self.reward_models.append(
                    LightningNetwork.load_from_checkpoint(
                        checkpoint_path=reward_model_path
                    )
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
            if self.is_expert_reward:
                rewards = torch.min(
                    torch.cat(
                        self.expert_model.policy.critic_target(  # type: ignore
                            torch.from_numpy(state).to(self.device),
                            torch.from_numpy(actions).to(self.device),
                        ),
                        dim=1,
                    ),
                    dim=1,
                )[0]
            else:
                rewards = torch.zeros(state.shape[0]).to(self.device)

                for reward_model in self.reward_models:
                    rewards += reward_model(
                        torch.cat(
                            [
                                torch.Tensor(state).to(self.device),
                                torch.Tensor(actions).to(self.device),
                            ],
                            dim=1,
                        )
                    ).squeeze(1)

                rewards /= len(self.reward_models)

        return rewards.cpu().numpy()


def main():
    """Run RL agent training."""

    script_path = Path(__file__).parents[1].resolve()
    checkpoints_path = path.join(script_path, "rl_checkpoints")

    cpu_count = os.cpu_count()
    cpu_count = cpu_count if cpu_count is not None else 8

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument(
        "--experiment", type=int, default=0, help="Experiment number"
    )
    arg_parser.add_argument(
        "--feedback_type",
        type=str,
        default="evaluative",
        help="Type of feedback to train the reward model",
    )
    arg_parser.add_argument(
        "--algorithm",
        type=str,
        default="sac",
        help="RL algorithm used to generate the feedback",
    )
    arg_parser.add_argument(
        "--environment",
        type=str,
        default="HalfCheetah-v3",
        help="Environment used to generate the feedback",
    )
    arg_parser.add_argument(
        "--use-sde",
        type=bool,
        default=False,
        help="Whether the RL algorithm used SDE",
    )
    arg_parser.add_argument(
        "--steps-per-checkpoint",
        type=int,
        default=10000,
        help="Number of steps per checkpoint",
    )
    arg_parser.add_argument(
        "--use-expert-reward",
        type=bool,
        default=False,
        help="Whether to use expert reward",
    )
    args = arg_parser.parse_args()

    # For PPO, the more environments there are, the more `num_timesteps` shifts
    # from `total_timesteps` (TODO: check this again)
    environment = make_vec_env(
        args.environment,
        n_envs=cpu_count if args.algorithm != "ppo" else 1,
        # n_envs=1,
        rng=random_generator,
    )

    FEEDBACK_ID = "_".join(
        [args.algorithm, args.environment, *(["sde"] if args.use_sde else [])]
    )
    MODEL_ID = f"#{args.experiment}_{FEEDBACK_ID}"

    IS_EXPERT_REWARD = args.use_expert_reward
    tensorboard_path = path.join(script_path, "rl_logs")

    reward_model_paths: list[str] = []

    # Select agent algorithm
    if args.algorithm == "sac":
        model_class = SAC
    elif args.algorithm == "ppo":
        model_class = PPO
    else:
        raise NotImplementedError(f"{args.algorithm} not implemented")

    REWARD_MODEL_ID = get_reward_model_name(
        MODEL_ID,
        args.feedback_type,
        args.use_reward_difference,
        args.experiment,
        feedback_override=(
            "without"
            if IS_EXPERT_REWARD
            else typing.cast(FeedbackType, args.feedback_type)
        ),
    )

    reward_model_paths.append(
        path.join(script_path, "reward_model_checkpoints", f"{REWARD_MODEL_ID}.ckpt")
    )

    RUN_NAME = get_reward_model_name(
        "-".join(sys.argv[1:]),
        args.feedback_type,
        args.use_reward_difference,
        args.experiment,
        feedback_override="without",
    )

    output_path = path.join(checkpoints_path, RUN_NAME)

    # if TRAINING_FEEDBACK_TYPE == "expert":
    # PPO
    # expert_model = PPO.load(
    #     path.join(
    #         script_path,
    #         "..",
    #         "experts",
    #         "ppo",
    #         "HalfCheetah-v3_1",
    #         "HalfCheetah-v3.zip",
    #     ),
    #     custom_objects={
    #         "learning_rate": 0.0,
    #         "lr_schedule": lambda _: 0.0,
    #         "clip_range": lambda _: 0.0,
    #     },
    # )

    # SAC
    if IS_EXPERT_REWARD:
        expert_model = model_class.load(
            path.join(
                script_path,
                "experts",
                args.algorithm,
                f"{args.environment}_1",
                f"{args.environment}.zip",
            ),
            custom_objects={
                "learning_rate": 0.0,
                "lr_schedule": lambda _: 0.0,
                "clip_range": lambda _: 0.0,
            },
        )

        environment = RewardVecEnvWrapper(
            environment,
            reward_fn=CustomReward(
                is_expert_reward=IS_EXPERT_REWARD,
                expert_model=expert_model,
            ),
        )

    if not IS_EXPERT_REWARD:

        print("Reward model ID:", MODEL_ID)
        environment = RewardVecEnvWrapper(
            environment,
            reward_fn=CustomReward(
                is_expert_reward=IS_EXPERT_REWARD,
                reward_model_paths=reward_model_paths,
                device=DEVICE,
                expert_model=expert_model,
            ),
        )

    print()

    # Select agent algorithm
    if args.algorithm == "sac":
        model_class = SAC
    elif args.algorithm == "ppo":
        model_class = PPO
    else:
        raise NotImplementedError(f"{args.algorithm} not implemented")

    model = model_class(
        "MlpPolicy",
        environment,
        # verbose=1,
        tensorboard_log=tensorboard_path,
        use_sde=args.use_sde,
        # gamma=0,
    )

    iterations = 30
    steps_per_iteration = 125000
    timesteps = 0

    # model.save(f"{output_path}_{timesteps}")

    for iteration_count in range(iterations):
        trained_model = model.learn(
            total_timesteps=steps_per_iteration * (iteration_count + 1) - timesteps,
            reset_num_timesteps=False,
            tb_log_name=RUN_NAME,
        )

        timesteps = trained_model.num_timesteps
        print(f"{timesteps}/{steps_per_iteration * iterations} steps done")

        model.save(f"{output_path}_{timesteps}")


if __name__ == "__main__":
    main()
