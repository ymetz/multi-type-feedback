from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import gymnasium as gym
import numpy as np
import torch
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor
from torch.utils.data import DataLoader

import wandb
from wandb.integration.sb3 import WandbCallback
from multi_type_feedback.feedback_dataset import BufferDataset, load_flat_buffer_into_feedback_dataset
from multi_type_feedback.feedback_oracle import FeedbackOracle
from multi_type_feedback.networks import (
    LightningCnnNetwork,
    LightningNetwork,
    calculate_pairwise_loss,
    calculate_single_reward_loss,
)
from multi_type_feedback.utils import TrainingUtils, get_project_root, RewardVecEnvWrapper

def one_hot_vector(k, max_val):
    vec = np.zeros(max_val)
    np.put(vec, k, 1)
    return vec

class DynamicRLHF:
    def __init__(
        self,
        oracle: FeedbackOracle,
        env: gym.Env,
        env_name: str = "Pendulum-v1",
        algorithm: str = "ppo",
        feedback_types: List[str] = [
            "evaluative",
            "comparative",
            "demonstrative",
            "descriptive",
        ],
        n_feedback_per_iteration: int = 50,
        feedback_buffer_size: int = 2000,
        rl_steps_per_iteration: int = 5000,
        reward_training_epochs: int = 2, # we train the rew. model after each update, just do one epoch
        device: str = "cuda",
        enable_wandb: bool = True,
        wandb_project_name: str = "dynamic_rlhf",
        num_ensemble_models: int = 4, # masksemble
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
        self.num_ensemble_models = num_ensemble_models
        self.action_one_hot = isinstance(self.env.action_space, gym.spaces.Discrete)
        if self.action_one_hot:
            self.one_hot_dim = self.env.action_space.n

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
                sync_tensorboard=True,
            )

            self.wandb_logger = WandbLogger(
                project=wandb_project_name,
                name=f"DYNAMIC_RL_{algorithm}_{env_name}_{','.join(feedback_types)}",
            )
        else:
            self.wandb_logger = None
        
    def _init_rl_agent(self):
        """Initialize the RL agent."""
        wrapped_env = RewardVecEnvWrapper(VecMonitor(DummyVecEnv([lambda: self.env])), reward_fn=self.compute_ensemble_reward)

        if self.algorithm == "ppo":
            return PPO("MlpPolicy", wrapped_env, verbose=1, device=self.device)
        else:
            return SAC("MlpPolicy", wrapped_env, verbose=1, device=self.device)

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
                action, _ = self.rl_agent.predict(obs, deterministic=False)
                next_obs, reward, terminated, truncated, _ = self.env.step(action)
                if self.action_one_hot:
                    action = one_hot_vector(action, self.one_hot_dim)
                done = terminated or truncated

                trajectory.append((np.expand_dims(obs, axis=0), action, reward, done))
                obs = next_obs

                if done:
                    break

            trajectories.append(trajectory)

        return trajectories, initial_states

    
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
            dataset = BufferDataset(
                buffer_data
            )
    
            # Train model
            trainer = Trainer(
                max_epochs=self.reward_training_epochs,
                accelerator="auto",
                devices="auto",
                enable_progress_bar=False,
                accumulate_grad_batches=32, # Virtual batch size 
                logger=self.wandb_logger,
                log_every_n_steps=10,
            )

            print("FB TYPE", feedback_type)
            
            trainer.fit(
                self.reward_models[feedback_type],
                DataLoader(
                    dataset,
                    batch_size=self.num_ensemble_models,
                    shuffle=True,
                    pin_memory=True,
                    drop_last=True
                ),
            )
    
            # Retrieve the final logged metrics for this model
            # "train_loss" will exist if you logged it with on_epoch=True
            final_metrics = trainer.callback_metrics
            
            reward_metrics[feedback_type] = float(final_metrics.get("train_loss", -1.0))

        return reward_metrics


    def compute_ensemble_reward(self, state: np.ndarray, action: np.ndarray) -> np.ndarray:
        device = self.device  # or whichever device you prefer

        if self.action_one_hot:
            action = one_hot_vector(action, self.one_hot_dim)

        # add batch dimension to actions if not present (if called with non-vectorized env.)
        if len(action.shape) < 2:
            action = np.expand_dims(action, axis=0)
    
        # Convert to torch tensors of shape [batch_size, ...]
        state_tensor = torch.as_tensor(state, device=device, dtype=torch.float32)
        action_tensor = torch.as_tensor(action, device=device, dtype=torch.float32)
    
        # Lists to accumulate each model's [batch_size,] reward and uncertainty
        model_rewards = []
        model_uncertainties = []
    
        with torch.no_grad():
            for feedback_type, reward_model in self.reward_models.items():
                # Only use models which have some feedback
                if len(self.feedback_buffers[feedback_type]) == 0:
                    continue
    
                if reward_model.ensemble_count > 1:
                    # Expand along a new "ensemble" dimension (dim=0)
                    # Resulting shape = [ensemble_count, batch_size, ...]
                    st_expanded = state_tensor.unsqueeze(0).expand(
                        reward_model.ensemble_count, *state_tensor.shape
                    )
                    act_expanded = action_tensor.unsqueeze(0).expand(
                        reward_model.ensemble_count, *action_tensor.shape
                    )
    
                    # predictions.shape might be [ensemble_count, batch_size]
                    # or [ensemble_count, batch_size, 1], etc.
                    predictions = reward_model(st_expanded, act_expanded)
    
                    # Make sure we reduce the final dimension if necessary
                    if predictions.dim() == 3 and predictions.shape[-1] == 1:
                        # e.g. shape: (ensemble_count, batch_size, 1)
                        predictions = predictions.squeeze(-1)
    
                    # Mean & std over the ensemble dimension (dim=0) => shape [batch_size,]
                    mean_reward = predictions.mean(dim=0)
                    uncertainty = predictions.std(dim=0)
                else:
                    # Single model in the ensemble
                    predictions = reward_model(state_tensor, action_tensor)
                    # e.g. shape: [batch_size] or [batch_size,1]
                    if predictions.dim() == 2 and predictions.shape[1] == 1:
                        predictions = predictions.squeeze(-1)
                    mean_reward = predictions
                    # Zero uncertainty
                    uncertainty = torch.zeros_like(mean_reward)
    
                # Collect
                model_rewards.append(mean_reward)         # shape [batch_size,]
                model_uncertainties.append(uncertainty)   # shape [batch_size,]
    
        # If no models have feedback, return zeros for the entire batch
        if not model_rewards:
            return np.zeros(state.shape[0], dtype=np.float32)
    
        # Stack across models => shape (#models, batch_size)
        stacked_rewards = torch.stack(model_rewards, dim=0)
        stacked_uncerts = torch.stack(model_uncertainties, dim=0)
    
        # final_rewards => shape [batch_size,]
        batch_size = state.shape[0]
        final_rewards = torch.zeros(batch_size, device=device, dtype=torch.float32)
    
        # Loop over each environment in the batch
        for i in range(batch_size):
            # For the i-th environment, gather all model rewards/uncertainties
            r_i = stacked_rewards[:, i]   # shape (#models,)
            u_i = stacked_uncerts[:, i]   # shape (#models,)
    
            if torch.any(u_i > 0):
                # If any model has a positive uncertainty, weight by 1 / uncertainty
                w_i = torch.where(u_i > 0, 1.0 / u_i, torch.ones_like(u_i))
                # Normalize weights
                w_i /= w_i.sum()
                final_rewards[i] = (r_i * w_i).sum()
            else:
                # Otherwise, just average over the models
                final_rewards[i] = r_i.mean()
    
        return final_rewards.cpu().numpy()  # shape: [batch_size,]


    def train_iteration(self, sampling_strategy: str = "random"):
        """Run one iteration of the training loop."""
        # Collect trajectories
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
        self.rl_agent.learn(
            total_timesteps=self.rl_steps_per_iteration, 
            reset_num_timesteps=False,
            callback=WandbCallback())

    def train(self, total_iterations: int, sampling_strategy: str = "random"):
        """Run full training loop for specified number of iterations."""
        for iteration in range(total_iterations):
            print(f"\nIteration {iteration + 1}/{total_iterations}")

            feedback_counts, reward_metrics = self.train_iteration(sampling_strategy)
            print(feedback_counts, reward_metrics)

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
    
    # Existing arguments
#

    TrainingUtils.set_seeds(args.seed)
    device = TrainingUtils.get_device()

    feedback_id, _ = TrainingUtils.get_model_ids(args)
    feedback_path = (
        Path(args.reference_data_folder) / f"{feedback_id}.pkl"
    )

    environment = TrainingUtils.setup_environment(args.environment, args.seed)
    expert_models = TrainingUtils.load_expert_models(
        env_name=args.environment,
        algorithm=args.algorithm,
        checkpoints_path=str(get_project_root() / args.expert_model_base_path),
        environment=environment,
        top_n_models=args.top_n_models,
    )

    # Initialize oracle
    oracle = FeedbackOracle(
        expert_models=expert_models,
        environment=environment,
        reference_data_path=feedback_path,
        noise_level=args.noise_level,
    )

    # Initialize RLHF trainer with CLI arguments
    rlhf = DynamicRLHF(
        oracle=oracle,
        env=environment,
        env_name=args.environment,
        algorithm=args.algorithm,
        feedback_types=args.feedback_types,
        n_feedback_per_iteration=args.n_feedback_per_iteration,
        feedback_buffer_size=args.feedback_buffer_size,
        rl_steps_per_iteration=args.rl_steps_per_iteration,
        reward_training_epochs=args.reward_training_epochs,
        device=device,
        enable_wandb=True,
        wandb_project_name=args.wandb_project_name,
        num_ensemble_models=args.num_ensemble_models,
    )

    # Run training with calculated iterations
    rlhf.train(
        total_iterations=total_iterations,
        sampling_strategy=args.sampling_strategy
    )

if __name__ == "__main__":
    main()