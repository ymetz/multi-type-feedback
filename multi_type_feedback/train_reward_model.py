"""Module for training a reward model from the generated feedback."""

import math
import os
from typing import Union

import torch
import wandb
from pytorch_lightning import Callback, LightningModule, Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import DataLoader, random_split

from multi_type_feedback.datatypes import FeedbackType
from multi_type_feedback.feedback_dataset import FeedbackDataset, LoadFeedbackDataset
from multi_type_feedback.networks import (
    SingleCnnNetwork,
    SingleNetwork,
    calculate_pairwise_loss,
    calculate_rtrank_pairwise_loss,
    calculate_single_reward_loss,
)
from multi_type_feedback.utils import TrainingUtils

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

# Utilize Tensor Cores of NVIDIA GPUs
torch.set_float32_matmul_precision("high")


def train_reward_model(
    reward_model: LightningModule,
    reward_model_id: str,
    feedback_type: FeedbackType,
    dataset: FeedbackDataset,
    maximum_epochs: int = 100,
    environment: str = "HalfCheetah-v5",
    gradient_clip_value: Union[float, None] = None,
    split_ratio: float = 0.8,
    enable_progress_bar=True,
    callback: Union[Callback, None] = None,
    num_ensemble_models: int = 4,
    batch_size_multiplier: int = 4,
    noise_level: float = 0.0,
    n_feedback: int = -1,
    seed: int = 0,
    wandb_project_name: str = "multi-type-rlhf",
    save_path: str = "reward_models",
    rt_loss_weight: float = 0.0,
    stratifier=None,
):
    """Train a reward model given trajectories data."""
    training_set_size = math.floor(split_ratio * len(dataset))
    train_set, val_set = random_split(
        dataset, lengths=[training_set_size, len(dataset) - training_set_size]
    )

    train_loader = DataLoader(
        train_set,
        batch_size=num_ensemble_models * batch_size_multiplier,
        shuffle=True,
        pin_memory=True,
        # num_workers=cpu_count,
        num_workers=0,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_set,
        batch_size=num_ensemble_models * batch_size_multiplier,
        pin_memory=True,
        num_workers=1,
        drop_last=True,
    )

    checkpoint_callback = ModelCheckpoint(
        dirpath=save_path,
        filename=reward_model_id,
        monitor="val_loss",
    )

    # initialise the wandb logger and name your wandb project
    wandb_logger = WandbLogger(
        project=wandb_project_name,
        name=reward_model_id,
        config={
            "feedback_type": feedback_type,
            "noise_level": noise_level,
            "seed": seed,
            "environment": environment,
            "n_feedback": n_feedback,
            "rt_loss_weight": rt_loss_weight,
            "stratifier": str(type(stratifier).__name__) if stratifier else None,
        },
    )

    trainer = Trainer(
        max_epochs=maximum_epochs,
        accelerator="auto",
        devices="auto",
        log_every_n_steps=5,
        gradient_clip_val=gradient_clip_value,
        enable_progress_bar=enable_progress_bar,
        logger=wandb_logger,
        # accumulate_grad_batches=4,
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
    parser.add_argument(
        "--feedback-folder",
        type=str,
        default="feedback",
        help="Folder to load feedback from",
    )
    parser.add_argument(
        "--save-folder",
        type=str,
        default="reward_models",
        help="Save folder for trained reward models",
    )
    parser.add_argument(
        "--rt-loss-weight",
        type=float,
        default=0.0,
        help="Weight for RT-rank loss component (0.0 = disabled)",
    )
    parser.add_argument(
        "--stratifier",
        type=str,
        default="none",
        choices=["knn", "std_window", "global", "none"],
        help="Stratification method for RT-rank loss",
    )
    args = parser.parse_args()

    TrainingUtils.set_seeds(args.seed)
    environment = TrainingUtils.setup_environment(
        args.environment, save_reset_wrapper=False
    )

    feedback_id, model_id = TrainingUtils.get_model_ids(args)

    # Setup stratifier if RT-rank loss is enabled
    stratifier = None
    if args.rt_loss_weight > 0.0:
        from multi_type_feedback.stratification import (
            GlobalPartitionStratifier,
            KnnStratifier,
            StdWindowStratifier,
        )
        
        if args.stratifier == "knn":
            stratifier = KnnStratifier(partition_clusters=8, min_cluster_size=4)
        elif args.stratifier == "std_window":
            stratifier = StdWindowStratifier(std_window=1.0, min_cluster_size=1)
        elif args.stratifier == "global":
            stratifier = GlobalPartitionStratifier(split_on_ties=False)
        elif args.stratifier == "none":
            stratifier = None
        else:
            raise ValueError(f"Unknown stratifier: {args.stratifier}")

    # Setup reward model
    reward_model = (
        SingleCnnNetwork
        if "procgen" in args.environment or "ALE" in args.environment
        else SingleNetwork
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
            else (calculate_rtrank_pairwise_loss if args.rt_loss_weight > 0.0 
                  else calculate_pairwise_loss)
        ),
        learning_rate=1e-5,
        ensemble_count=args.n_ensemble,
        rt_loss_weight=args.rt_loss_weight,
    )

    dataset = LoadFeedbackDataset(
        os.path.join(args.feedback_folder, f"{feedback_id}.pkl"),
        args.feedback_type,
        args.n_feedback,
        noise_level=args.noise_level,
        env=environment if args.feedback_type == "demonstrative" else None,
        env_name=args.environment,
        seed=args.seed,
        discount_factor=discount_factors.get(
            args.environment, 0.99
        ),  # adapt for custom envs
        stratifier=stratifier,
    )

    train_reward_model(
        reward_model,
        model_id,
        args.feedback_type,
        dataset,
        maximum_epochs=100,
        split_ratio=0.85,
        environment=args.environment,
        num_ensemble_models=args.n_ensemble,
        enable_progress_bar=not args.no_loading_bar,
        noise_level=args.noise_level,
        n_feedback=args.n_feedback,
        seed=args.seed,
        wandb_project_name=args.wandb_project_name,
        save_path=args.save_folder,
        rt_loss_weight=args.rt_loss_weight,
        stratifier=stratifier,
    )


if __name__ == "__main__":
    main()
