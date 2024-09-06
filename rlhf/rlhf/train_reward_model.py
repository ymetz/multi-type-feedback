"""Module for training a reward model from the generated feedback."""

import argparse
import math
import os
import pickle
from os import path
from pathlib import Path
from random import randint, randrange
from typing import Union

import numpy
import torch
from pytorch_lightning import Callback, LightningModule, Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import DataLoader, Dataset, random_split

import wandb
from rlhf.common import get_reward_model_name
from rlhf.datatypes import Feedback, FeedbackType
from rlhf.networks import LightningNetwork, calculate_mle_loss, calculate_mse_loss

script_path = Path(__file__).parents[1].resolve()

# Utilize Tensor Cores of NVIDIA GPUs
torch.set_float32_matmul_precision("high")


class FeedbackDataset(Dataset):
    """PyTorch Dataset for loading the feedback data."""

    def __init__(
        self,
        dataset_path: str,
        feedback_type: FeedbackType,
        use_reward_difference: bool,
        steps_per_checkpoint: int,
    ):
        """Initialize dataset."""
        print("Loading dataset...")

        with open(dataset_path, "rb") as feedback_file:
            feedback_list: list[Feedback] = pickle.load(feedback_file)

        expert_value_key = (
            "expert_value" if not use_reward_difference else "expert_value_difference"
        )

        match feedback_type:
            case "evaluative":
                # First: Observation, Second: Reward
                self.first = [
                    numpy.concatenate(
                        [feedback["observation"], feedback["actions"]]
                    ).astype("float32")
                    for feedback in feedback_list
                ]
                self.second = [
                    numpy.float32(feedback[expert_value_key])
                    for feedback in feedback_list
                ]
            case "comparative":
                # First: high-reward observation, Second: low-reward observation
                observation_pairs = [
                    map(
                        lambda feedback: numpy.concatenate(
                            [feedback["observation"], feedback["actions"]]
                        ).astype("float32"),
                        sorted(
                            list(
                                (
                                    feedback_list[randint(0, len(feedback_list) - 1)],
                                    feedback_list[randint(0, len(feedback_list) - 1)],
                                )
                            ),
                            key=lambda feedback: feedback[expert_value_key],
                            reverse=True,
                        ),
                    )
                    for _ in range(len(feedback_list))
                ]

                self.first, self.second = zip(*observation_pairs)
            case "corrective" | "demonstrative":
                demonstrative_length = (
                    steps_per_checkpoint if feedback_type == "demonstrative" else None
                )

                # First: Expert's observation, Second: Agent's observation
                # Note: this only works for IRL (for observation-only models, use
                # `expert_observation` and `next_observation` for the first and second respectively)
                # TODO: experiment with changing the threshold
                self.first = [
                    numpy.concatenate(
                        [feedback["expert_observation"], feedback["expert_actions"]]
                    ).astype("float32")
                    for feedback in feedback_list
                    if feedback["expert_own_value"] > feedback["expert_value"]
                ][:demonstrative_length]

                self.second = [
                    numpy.concatenate(
                        [feedback["observation"], feedback["actions"]]
                    ).astype("float32")
                    for feedback in feedback_list
                    if feedback["expert_own_value"] > feedback["expert_value"]
                ][:demonstrative_length]
            case "descriptive":
                # First: Changed observation, Second: Agent's observation
                # TODO: generate more perturbation for one feedback
                model_inputs = numpy.array(
                    list(
                        map(
                            lambda feedback: numpy.concatenate(
                                [feedback["observation"], feedback["actions"]]
                            ).astype("float32"),
                            feedback_list,
                        )
                    )
                )

                standard_deviations = numpy.std(model_inputs, axis=0)

                for index, feedback in enumerate(feedback_list):
                    perturbations = numpy.random.normal(
                        0, standard_deviations, model_inputs.shape[-1]
                    )

                    perturbations[feedback["expert_value_attributions"] > 0] = 0

                    model_inputs[index] += perturbations

                # First: Observation, Second: Reward
                self.first = model_inputs
                self.second = [
                    numpy.float32(feedback[expert_value_key])
                    for feedback in feedback_list
                ]
            case _:
                raise NotImplementedError(
                    "Dataset not implemented for this feedback type ."
                )

        print("Dataset loaded")

    def __len__(self):
        """Return size of dataset."""
        return len(self.first)

    def __getitem__(self, index):
        """Return item with given index."""
        return self.first[index], self.second[index]


def train_reward_model(
    reward_model: LightningModule,
    reward_model_id: str,
    feedback_type: FeedbackType,
    use_reward_difference: bool,
    dataset: FeedbackDataset,
    maximum_epochs: int,
    batch_size: int,
    cpu_count: int = 4,
    algorithm: str = "sac",
    environment: str = "HalfCheetah-v3",
    use_sde: bool = False,
    gradient_clip_value: Union[float, None] = None,
    split_ratio: float = 0.8,
    enable_progress_bar=True,
    callback: Union[Callback, None] = None,
):

    get_reward_model_name(
        reward_model_id,
        feedback_type,
        use_reward_difference,
        f"{randrange(1000, 10000)}",
    )

    """Train a reward model given trajectories data."""
    training_set_size = math.floor(split_ratio * len(dataset))
    train_set, val_set = random_split(
        dataset, lengths=[training_set_size, len(dataset) - training_set_size]
    )

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=True,
        num_workers=cpu_count,
    )

    val_loader = DataLoader(
        val_set, batch_size=batch_size, pin_memory=True, num_workers=cpu_count
    )

    checkpoint_callback = ModelCheckpoint(
        dirpath=path.join(script_path, "..", "reward_model_checkpoints"),
        filename=reward_model_id,
        monitor="val_loss",
    )

    # initialise the wandb logger and name your wandb project
    wandb_logger = WandbLogger(project="Multi-Feedback-RLHF", name=reward_model_id)

    # add your batch size to the wandb config
    wandb_logger.experiment.config.update(
        {
            "rl_algorithm": algorithm,
            "rl_environment": environment,
            "rl_is_use_sde": use_sde,
            "rl_feedback_type": feedback_type,
            "max_epochs": maximum_epochs,
            "batch_size": batch_size,
            "gradient_clip_value": gradient_clip_value,
            "learning_rate": reward_model.learning_rate,
        }
    )

    trainer = Trainer(
        max_epochs=maximum_epochs,
        log_every_n_steps=5,
        gradient_clip_val=gradient_clip_value,
        enable_progress_bar=enable_progress_bar,
        logger=wandb_logger,
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
    args = arg_parser.parse_args()

    FEEDBACK_ID = "_".join(
        [args.algorithm, args.environment, *(["sde"] if args.use_sde else [])]
    )
    MODEL_ID = f"#{args.experiment}_{FEEDBACK_ID}"

    # Load data
    dataset = FeedbackDataset(
        path.join(script_path, "feedback", f"{FEEDBACK_ID}.pkl"),
        args.feedback_type,
        args.use_reward_difference,
        args.steps_per_checkpoint,
    )

    # Select loss function based on feedback type
    loss_function = None

    match args.feedback_type:
        case "evaluative" | "descriptive":
            loss_function = calculate_mse_loss
        case "comparative" | "corrective" | "demonstrative":
            loss_function = calculate_mle_loss
        case _:
            raise NotImplementedError(
                "Loss function not implemented for this feedback type."
            )

    # Train reward model
    reward_model = LightningNetwork(
        input_dim=23,
        hidden_dim=256,
        layer_num=12,
        output_dim=1,
        loss_function=loss_function,
        learning_rate=(
            1e-6
            if args.feedback_type == "corrective"
            else (1e-5 if args.feedback_type == "comparative" else 2e-5)
        ),
    )

    train_reward_model(
        reward_model,
        MODEL_ID,
        args.feedback_type,
        args.use_reward_difference,
        dataset,
        maximum_epochs=100,
        batch_size=4,
        split_ratio=0.5,
        cpu_count=cpu_count,
    )


if __name__ == "__main__":
    main()
