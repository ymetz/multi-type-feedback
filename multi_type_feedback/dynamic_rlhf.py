import argparse
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import gymnasium as gym
import numpy as np
import torch
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping
from stable_baselines3 import PPO, SAC
from torch.utils.data import DataLoader

import wandb
from multi_type_feedback.feedback_dataset import FeedbackDataset, load_flat_buffer_into_feedback_dataset
from multi_type_feedback.feedback_oracle import FeedbackOracle
from multi_type_feedback.networks import (
    LightningCnnNetwork,
    LightningNetwork,
    calculate_pairwise_loss,
    calculate_single_reward_loss,
)
from multi_type_feedback.utils import TrainingUtils


class DynamicRLHF:
    def __init__(
        self,
        oracle: FeedbackOracle,
        env: gym.Env,
        algorithm: str = "ppo",
        feedback_types: List[str] = [
            "evaluative",
            "comparative",
            "demonstrative",
            "descriptive",
        ],
        n_feedback_per_iteration: int = 100,
        feedback_buffer_size: int = 1000,
        rl_steps_per_iteration: int = 10000,
        reward_training_epochs: int = 50,
        device: str = "cuda",
        enable_wandb: bool = True,
        wandb_project_name: str = "dynamic_rlhf",
    ):
        self.oracle = oracle
        self.env = env
        self.algorithm = algorithm
        self.feedback_types = feedback_types
        self.n_feedback_per_iteration = n_feedback_per_iteration
        self.feedback_buffer_size = feedback_buffer_size
        self.rl_steps_per_iteration = rl_steps_per_iteration
        self.reward_training_epochs = reward_training_epochs
        self.device = device
        self.enable_wandb = enable_wandb

        # Initialize feedback buffers for each type
        self.feedback_buffers = {feedback_type: [] for feedback_type in feedback_types}

        # Initialize RL agent
        self.rl_agent = self._init_rl_agent()

        # Initialize reward models for each feedback type
        self.reward_models = self._init_reward_models()

        if enable_wandb:
            wandb.init(
                project=wandb_project_name,
                config={
                    "algorithm": algorithm,
                    "feedback_types": feedback_types,
                    "n_feedback_per_iteration": n_feedback_per_iteration,
                    "rl_steps_per_iteration": rl_steps_per_iteration,
                },
            )

    def _init_rl_agent(self):
        """Initialize the RL agent."""
        if self.algorithm == "ppo":
            return PPO("MlpPolicy", self.env, verbose=1, device=self.device)
        else:
            return SAC("MlpPolicy", self.env, verbose=1, device=self.device)

    def _init_reward_models(self):
        """Initialize reward models for each feedback type."""
        reward_models = {}

        for feedback_type in self.feedback_types:
            if "ALE/" in self.env.spec.id or "procgen" in self.env.spec.id:
                model = LightningCnnNetwork(
                    input_spaces=(self.env.observation_space, self.env.action_space),
                    hidden_dim=256,
                    action_hidden_dim=16,
                    layer_num=3,
                    cnn_channels=(16, 32, 32),
                    output_dim=1,
                    loss_function=(
                        calculate_single_reward_loss
                        if feedback_type in ["evaluative", "descriptive"]
                        else calculate_pairwise_loss
                    ),
                    learning_rate=1e-5,
                    ensemble_count=4,
                )
            else:
                model = LightningNetwork(
                    input_spaces=(self.env.observation_space, self.env.action_space),
                    hidden_dim=256,
                    action_hidden_dim=32,
                    layer_num=6,
                    output_dim=1,
                    loss_function=(
                        calculate_single_reward_loss
                        if feedback_type in ["evaluative", "descriptive"]
                        else calculate_pairwise_loss
                    ),
                    learning_rate=1e-5,
                    ensemble_count=4,
                )
            reward_models[feedback_type] = model

        return reward_models

    def collect_trajectories(self, n_trajectories: int) -> List[Dict]:
        """Collect trajectories using current policy."""
        trajectories = []
        initial_states = []

        for _ in range(n_trajectories):
            trajectory = []
            obs, _ = self.env.reset()
            initial_states.append(self.env.save_state(observation=obs))

            for _ in range(self.oracle.segment_len):
                action, _ = self.rl_agent.predict(obs, deterministic=True)
                next_obs, reward, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated

                trajectory.append((np.expand_dims(obs, axis=0), action, reward, done))
                obs = next_obs

                if done:
                    break

            trajectories.append(trajectory)

        return trajectories, initial_states

    def update_feedback_buffers(self, new_feedback: List[Dict]):
        """Update feedback buffers with new feedback while maintaining size limit."""
        for feedback in new_feedback:
            for feedback_type in feedback:
                if feedback_type != "uncertainty":  # Skip uncertainty metadata
                    if (
                        len(self.feedback_buffers[feedback_type])
                        >= self.feedback_buffer_size
                    ):
                        # Remove oldest feedback
                        self.feedback_buffers[feedback_type].pop(0)
                    self.feedback_buffers[feedback_type].append(feedback[feedback_type])

    def compute_model_uncertainty(
        self,
        trajectory: List[Tuple[np.ndarray, np.ndarray, float, bool]],
        feedback_type: str,
    ) -> float:
        """Compute uncertainty for a trajectory using the ensemble variance of the specific reward model."""
        reward_model = self.reward_models[feedback_type]

        # Stack observations and actions from trajectory
        states = torch.vstack(
            [torch.as_tensor(step[0]).float() for step in trajectory]
        ).to(self.device)
        actions = torch.vstack(
            [torch.as_tensor(step[1]).float() for step in trajectory]
        ).to(self.device)

        # Get predictions from all ensemble members
        with torch.no_grad():
            if reward_model.ensemble_count > 1:
                states_expanded = states.unsqueeze(0).expand(
                    reward_model.ensemble_count, *states.shape
                )
                actions_expanded = actions.unsqueeze(0).expand(
                    reward_model.ensemble_count, *actions.shape
                )
                predictions = reward_model(
                    states_expanded, actions_expanded
                )  # Shape: [ensemble_size, traj_len, 1]

                # Compute trajectory-level uncertainty as mean of step-wise uncertainties
                step_uncertainties = predictions.std(
                    dim=0
                )  # Standard deviation across ensemble members
                trajectory_uncertainty = (
                    step_uncertainties.mean().item()
                )  # Mean uncertainty across trajectory
            else:
                trajectory_uncertainty = 0.0

        return trajectory_uncertainty

    def sample_feedback_uncertainty(
        self, trajectories: List[List], initial_states: List[np.ndarray]
    ) -> Dict:
        """Sample feedback types based on ensemble variance for each reward model."""
        # Calculate uncertainties for each trajectory and feedback type
        trajectory_uncertainties = []

        for trajectory in trajectories:
            uncertainties = {}
            for feedback_type in self.feedback_types:
                if (
                    len(self.feedback_buffers[feedback_type]) > 0
                ):  # Only if model has been trained
                    uncertainty = self.compute_model_uncertainty(
                        trajectory, feedback_type
                    )
                else:
                    # If no feedback yet, set high uncertainty to encourage exploration
                    uncertainty = float("inf")
                uncertainties[feedback_type] = uncertainty
            trajectory_uncertainties.append(uncertainties)

        # Sample feedback types based on uncertainties
        feedback_counts = defaultdict(int)
        all_feedback = []

        # For each trajectory, sample feedback type with probability proportional to uncertainty
        for trajectory, initial_state, uncertainties in zip(
            trajectories, initial_states, trajectory_uncertainties
        ):
            # Normalize uncertainties to probabilities
            total_uncertainty = sum(uncertainties.values())
            if total_uncertainty == float("inf"):
                # If no feedback yet for some types, sample uniformly from those
                untrained_types = [
                    ft
                    for ft in self.feedback_types
                    if len(self.feedback_buffers[ft]) == 0
                ]
                feedback_type = np.random.choice(untrained_types)
            else:
                probs = [
                    uncertainties[ft] / total_uncertainty for ft in self.feedback_types
                ]
                feedback_type = np.random.choice(self.feedback_types, p=probs)

            # Get feedback for selected type
            feedback = self.oracle.get_feedback(
                trajectory, initial_state, [feedback_type]
            )
            feedback["selected_uncertainty"] = uncertainties[
                feedback_type
            ]  # Store for logging

            feedback_counts[feedback_type] += 1
            all_feedback.append(feedback)

        return all_feedback, feedback_counts

    def train_iteration(self, sampling_strategy: str = "random"):
        """Run one iteration of the training loop."""
        # Collect trajectories
        trajectories, initial_states = self.collect_trajectories(
            self.n_feedback_per_iteration
        )

        # Get feedback based on sampling strategy
        if sampling_strategy == "random":
            feedback, feedback_counts = self.sample_feedback_random(
                trajectories, initial_states
            )
        else:  # uncertainty
            feedback, feedback_counts = self.sample_feedback_uncertainty(
                trajectories, initial_states
            )

        # Update feedback buffers
        self.update_feedback_buffers(feedback)

        # Ensure buffer data is in correct format before creating dataset
        for feedback_type, buffer in self.feedback_buffers.items():
            if buffer:
                self.feedback_buffers[feedback_type] = [
                    (obs, label, weight) 
                    for obs, label, weight in buffer
                ]

        # Train reward models
        reward_metrics = self.train_reward_models()

        # Calculate mean uncertainties for logging
        mean_uncertainties = defaultdict(list)
        if sampling_strategy == "uncertainty":
            for f in feedback:
                if "selected_uncertainty" in f:
                    mean_uncertainties["selected_uncertainty"].append(
                        f["selected_uncertainty"]
                    )

        # Train RL agent with updated reward models
        self.train_rl_agent()

        # Log metrics
        if self.enable_wandb:
            metrics = {"feedback_counts": feedback_counts, **reward_metrics}

            # Add uncertainty metrics if available
            if mean_uncertainties:
                metrics.update(
                    {
                        "mean_selected_uncertainty": np.mean(
                            mean_uncertainties["selected_uncertainty"]
                        ),
                        "max_selected_uncertainty": np.max(
                            mean_uncertainties["selected_uncertainty"]
                        ),
                        "min_selected_uncertainty": np.min(
                            mean_uncertainties["selected_uncertainty"]
                        ),
                    }
                )

            wandb.log(metrics)

        return feedback_counts, reward_metrics

    def sample_feedback_random(
        self, trajectories: List[List], initial_states: List[np.ndarray]
    ) -> Dict:
        """Randomly sample feedback types."""
        feedback_distribution = np.ones(len(self.feedback_types)) / len(
            self.feedback_types
        )
        selected_types = np.random.choice(
            self.feedback_types,
            size=self.n_feedback_per_iteration,
            p=feedback_distribution,
        )

        feedback_counts = defaultdict(int)
        all_feedback = []

        for trajectory, initial_state, feedback_type in zip(
            trajectories, initial_states, selected_types
        ):
            feedback_dict = {}
            
            # Handle different feedback types
            if feedback_type in ["comparative", "descriptive_preference"]:
                # Need a second trajectory for comparison
                trajectory2, _ = self.collect_trajectories(1)
                feedback = self.oracle.get_feedback(
                    (trajectory, trajectory2[0]), initial_state, feedback_type
                )
            elif feedback_type in ["demonstrative", "corrective"]:
                # Oracle will generate the second trajectory
                feedback = self.oracle.get_feedback(
                    trajectory, initial_state, feedback_type
                )
            else:  # evaluative, descriptive
                feedback = self.oracle.get_feedback(
                    trajectory, initial_state, feedback_type
                )
            
            feedback_dict[feedback_type] = feedback
            feedback_counts[feedback_type] += 1
            all_feedback.append(feedback_dict)

        return all_feedback, feedback_counts

    def update_feedback_buffers(self, new_feedback: List[Dict]):
        """Update feedback buffers with new feedback while maintaining size limit."""
        for feedback_dict in new_feedback:
            for feedback_type, feedback in feedback_dict.items():
                if feedback_type != "uncertainty":  # Skip uncertainty metadata
                    if len(self.feedback_buffers[feedback_type]) >= self.feedback_buffer_size:
                        # Remove oldest feedback
                        self.feedback_buffers[feedback_type].pop(0)
                    self.feedback_buffers[feedback_type].append(feedback)

    def train_reward_models(self):
        reward_metrics = {}
        
        for feedback_type in self.feedback_types:
            buffer_data = self.feedback_buffers[feedback_type]
            if not buffer_data:
                continue
                
            # Create dataset from buffer
            dataset = FeedbackDataset(
                load_flat_buffer_into_feedback_dataset(buffer_data, feedback_type),
                feedback_type,
                len(self.feedback_buffers[feedback_type]),
                env_name=self.env.spec.id,
                noise_level=0.0,  # No additional noise during training
                segment_len=self.oracle.segment_len,
                env=self.env if feedback_type == "demonstrative" else None,
            )

            # Train model
            trainer = Trainer(
                max_epochs=self.reward_training_epochs,
                devices=[0] if self.device == "cuda" else None,
                enable_progress_bar=False,
                logger=None,  # We'll handle logging manually
                callbacks=[EarlyStopping(monitor="val_loss", mode="min", patience=5)],
            )

            trainer.fit(
                self.reward_models[feedback_type],
                DataLoader(dataset, batch_size=32, shuffle=True),
            )

            metrics[f"{feedback_type}_loss"] = trainer.callback_metrics[
                "val_loss"
            ].item()

        return metrics

    def compute_ensemble_reward(self, state: np.ndarray, action: np.ndarray) -> float:
        """Compute reward using ensemble of reward models."""
        rewards = []
        uncertainties = []

        state_tensor = torch.as_tensor(
            state, device=self.device, dtype=torch.float32
        ).unsqueeze(0)
        action_tensor = torch.as_tensor(
            action, device=self.device, dtype=torch.float32
        ).unsqueeze(0)

        with torch.no_grad():
            for feedback_type, reward_model in self.reward_models.items():
                if (
                    len(self.feedback_buffers[feedback_type]) > 0
                ):  # Only use models with feedback
                    if reward_model.ensemble_count > 1:
                        state_expanded = state_tensor.expand(
                            reward_model.ensemble_count, *state_tensor.shape[1:]
                        )
                        action_expanded = action_tensor.expand(
                            reward_model.ensemble_count, *action_tensor.shape[1:]
                        )
                        predictions = reward_model(state_expanded, action_expanded)
                        mean_reward = predictions.mean().item()
                        uncertainty = predictions.std().item()
                    else:
                        predictions = reward_model(state_tensor, action_tensor)
                        mean_reward = predictions.item()
                        uncertainty = 0.0

                    rewards.append(mean_reward)
                    uncertainties.append(uncertainty)

        if not rewards:  # If no models have feedback yet
            return 0.0

        # Weight rewards by inverse uncertainty
        if any(u > 0 for u in uncertainties):
            weights = [1 / u if u > 0 else 1.0 for u in uncertainties]
            weights = [w / sum(weights) for w in weights]
            final_reward = sum(r * w for r, w in zip(rewards, weights))
        else:
            final_reward = sum(rewards) / len(rewards)

        return final_reward

    def train_iteration(self, sampling_strategy: str = "random"):
        """Run one iteration of the training loop."""
        # Collect trajectories
        trajectories, initial_states = self.collect_trajectories(
            self.n_feedback_per_iteration
        )

        # Get feedback based on sampling strategy
        if sampling_strategy == "random":
            feedback, feedback_counts = self.sample_feedback_random(
                trajectories, initial_states
            )
        else:  # uncertainty
            feedback, feedback_counts = self.sample_feedback_uncertainty(
                trajectories, initial_states
            )

        # Update feedback buffers
        self.update_feedback_buffers(feedback)

        # Train reward models
        reward_metrics = self.train_reward_models()

        # Train RL agent with updated reward models
        self.train_rl_agent()

        # Log metrics
        if self.enable_wandb:
            metrics = {"feedback_counts": feedback_counts, **reward_metrics}
            wandb.log(metrics)

        return feedback_counts, reward_metrics

    def train_rl_agent(self):
        """Train RL agent using current reward models."""

        # Create a reward wrapper for the environment
        class RewardWrapper(gym.Wrapper):
            def __init__(self, env, reward_fn):
                super().__init__(env)
                self.reward_fn = reward_fn

            def step(self, action):
                obs, _, terminated, truncated, info = super().step(action)
                reward = self.reward_fn(obs, action)
                return obs, reward, terminated, truncated, info

        # Wrap environment with ensemble reward
        wrapped_env = RewardWrapper(self.env, self.compute_ensemble_reward)

        # Update environment reference in RL agent
        self.rl_agent.env = wrapped_env

        # Train for specified number of steps
        self.rl_agent.learn(total_timesteps=self.rl_steps_per_iteration)

    def train(self, total_iterations: int, sampling_strategy: str = "random"):
        """Run full training loop for specified number of iterations."""
        for iteration in range(total_iterations):
            print(f"\nIteration {iteration + 1}/{total_iterations}")

            feedback_counts, reward_metrics = self.train_iteration(sampling_strategy)

            # Print progress
            print("\nFeedback counts:")
            for feedback_type, count in feedback_counts.items():
                print(f"{feedback_type}: {count}")

            print("\nReward model losses:")
            for feedback_type, loss in reward_metrics.items():
                print(f"{feedback_type}: {loss:.4f}")

        if self.enable_wandb:
            wandb.finish()


def main():
    parser = TrainingUtils.setup_base_parser()
    parser.add_argument(
        "--feedback-types",
        nargs="+",
        type=str,
        default=["evaluative", "comparative", "demonstrative", "descriptive"],
        help="Types of feedback to use",
    )
    parser.add_argument(
        "--sampling-strategy",
        type=str,
        default="random",
        choices=["random", "uncertainty"],
        help="Feedback sampling strategy",
    )
    parser.add_argument(
        "--save-folder", type=str, default="feedback", help="Save folder"
    )
    parser.add_argument(
        "--total-iterations",
        type=int,
        default=100,
        help="Number of training iterations",
    )
    parser.add_argument(
        "--top-n-models", type=int, default=3, help="Top N models to use"
    )
    args = parser.parse_args()

    TrainingUtils.set_seeds(args.seed)
    device = TrainingUtils.get_device()

    feedback_id, _ = TrainingUtils.get_model_ids(args)
    feedback_path = (
        Path(__file__).parents[1].resolve() / args.save_folder / f"{feedback_id}.pkl"
    )

    environment = TrainingUtils.setup_environment(args.environment, args.seed)
    expert_models = TrainingUtils.load_expert_models(
        args.environment,
        args.algorithm,
        "../main/gt_agents",
        environment,
        args.top_n_models,
    )

    # Initialize oracle
    oracle = FeedbackOracle(
        expert_models=expert_models,
        environment=environment,
        reference_data_path=feedback_path,
    )

    # Initialize RLHF trainer
    rlhf = DynamicRLHF(
        oracle=oracle,
        env=environment,
        algorithm=args.algorithm,
        feedback_types=args.feedback_types,
        wandb_project_name=args.wandb_project_name,
    )

    # Run training
    rlhf.train(
        total_iterations=args.total_iterations, sampling_strategy=args.sampling_strategy
    )


if __name__ == "__main__":
    main()
