"""Script to plot the reward model output against the ground truth values."""

import pickle
import sys
from os import path
from pathlib import Path

import matplotlib
import numpy
import torch
from matplotlib import pyplot
from torch import Tensor
from rlhf.common import get_reward_model_name
import argparse
from .datatypes import Feedback
from .networks import LightningNetwork

# Uncomment line below to use PyPlot with VSCode Tunnels
matplotlib.use("agg")

def main():
    """Plot reward model output."""

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
        "--use-reward-difference",
        type=bool,
        default=False,
        help="Whether to use the reward difference",
    )
    arg_parser.add_argument(
        "--steps-per-checkpoint",
        type=int,
        default=10000,
        help="Number of steps per checkpoint",
    )
    arg_parser.add_argument(
        "--checkpoint-number",
        type=int,
        default=0,
        help="Checkpoint number",
    )
    arg_parser.add_argument(
        "--num-steps",
        type=int,
        default=1000,
        help="Number of steps to plot",
    )
    args = arg_parser.parse_args()

    FEEDBACK_ID = "_".join(
        [args.algorithm, args.environment, *(["sde"] if args.use_sde else [])]
    )

    REWARD_MODEL_ID = get_reward_model_name(sys.argv[1], "evaluative", False, 0)

    script_path = Path(__file__).parent.resolve()

    feedback_path = path.join(script_path, "..", "feedback", f"{FEEDBACK_ID}.pkl")
    reward_model_path = path.join(
        script_path, "..", "reward_model_checkpoints", f"{REWARD_MODEL_ID}.ckpt"
    )

    output_path = path.join(script_path, "..", "plots", "reward_model_output.png")

    print("Feedback ID:", FEEDBACK_ID)
    print("Model ID:", REWARD_MODEL_ID)
    print()

    with open(feedback_path, "rb") as feedback_file:
        feedback_list: list[Feedback] = pickle.load(feedback_file)

    # pylint: disable=no-value-for-parameter
    reward_model = LightningNetwork.load_from_checkpoint(reward_model_path)

    feedback_start = args.steps_per_checkpoint * args.checkpoint_number
    feedback_end = feedback_start + args.num_steps

    observations = list(map(lambda feedback: feedback["observation"], feedback_list))[
        feedback_start:feedback_end
    ]

    actions = list(map(lambda feedback: feedback["actions"], feedback_list))[
        feedback_start:feedback_end
    ]

    rewards = list(map(lambda feedback: feedback["reward"], feedback_list))[
        feedback_start:feedback_end
    ]

    expert_value_predictions = list(
        map(lambda feedback: feedback["expert_value"], feedback_list)
    )[feedback_start:feedback_end]

    predicted_rewards = []

    steps = range(args.num_steps)

    observation_tensor = Tensor(numpy.array(observations)).to(DEVICE)
    actions_tensor = Tensor(numpy.array(actions)).to(DEVICE)

    print("Predicting rewards...")

    for i in steps:
        predicted_rewards.append(
            reward_model(torch.cat([observation_tensor[i], actions_tensor[i]]))
            .detach()
            .cpu()
        )

        if (i + 1) % 1000 == 0:
            print(f"{i + 1}/{args.num_steps}, done")
            print(
                f"difference: {predicted_rewards[-1] - expert_value_predictions[i]}\n"
            )

    print()

    pyplot.plot(steps, predicted_rewards, label="Reward model")
    # pyplot.plot(steps, expert_value_predictions, label="Expert value")
    pyplot.plot(steps, rewards, label="Ground truth rewards")

    pyplot.xlabel("Steps")
    pyplot.ylabel("Rewards")
    pyplot.legend()

    pyplot.savefig(output_path)


if __name__ == "__main__":
    main()
