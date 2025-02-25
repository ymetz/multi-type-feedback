from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple, Any

import gymnasium as gym
import numpy as np
import torch
import uuid
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor
from stable_baselines3.common.callbacks import BaseCallback
from train_baselines.exp_manager import ExperimentManager
from torch.utils.data import DataLoader

import wandb
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

def vectorized_one_hot_vector(k, max_val):
    vec = np.zeros((k.size, max_val))
    vec[np.arange(k.size), k] = 1
    return vec

def create_matrix_vectorized(rows, cols, step):
    """
    Create a matrix that can be used to vectorize averaging over models
    """
    idx = torch.arange(0, cols).view(1, -1)
    row_indices = torch.arange(0, rows).view(-1, 1)
    matrix = (idx - row_indices) % step == 0
    return matrix.float()

def compute_grouped(tensor, k):
    """
    Compute standard deviation for groups of elements spaced k apart.
    
    Args:
        tensor: Input tensor of shape (N,) where N is divisible by k
        k: Number of predictions per input
    
    Returns:
        Tensor of shape (N//k,) containing standard deviations
    """
    # Reshape the tensor to group related predictions together
    n_inputs = tensor.shape[0] // k
    reshaped = tensor.reshape(k, n_inputs).t()  # Shape: (n_inputs, k)
    
    # Compute standard deviation along dimension 1 (across the k predictions)
    return torch.mean(reshaped, dim=1), torch.std(reshaped, dim=1)  # Shape: (n_inputs,)

class DynamicRLHF:
    def __init__(
        self,
        oracle: FeedbackOracle,
        env: gym.Env,
        gen_env: gym.Env,
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
        reward_training_epochs: int = 2,
        device: str = "cuda",
        num_ensemble_models: int = 4,
        callbacks: List[BaseCallback] = None,
        hyperparams: Dict[str, Any] = None,  # Hyperparameters from ExperimentManager
        seed: int = None,
        wandb_logger: Any = None,
        tensorboard_log: str = "",
    ):
        self.oracle = oracle
        self.env = env
        self.gen_env = gen_env
        self.env_name = env_name
        self.algorithm = algorithm
        self.feedback_types = feedback_types
        self.n_feedback_per_iteration = n_feedback_per_iteration
        self.feedback_buffer_size = feedback_buffer_size
        self.rl_steps_per_iteration = rl_steps_per_iteration
        self.reward_training_epochs = reward_training_epochs
        self.device = device
        self.num_ensemble_models = num_ensemble_models
        self.callbacks = callbacks or []
        self._hyperparams = hyperparams or {}
        self.seed = seed
        self.wandb_logger = wandb_logger
        self.tensorboard_log = tensorboard_log

        unique_id = str(uuid.uuid4())[:8]
        self.tb_log_name = f"tb_log_{unique_id}"
        
        self.action_one_hot = isinstance(self.env.action_space, gym.spaces.Discrete)
        if self.action_one_hot:
            self.one_hot_dim = self.env.action_space.n

        # Initialize feedback buffers for each type
        self.feedback_buffers = {feedback_type: [] for feedback_type in feedback_types}

        # Initialize RL agent
        self.rl_agent = self._init_rl_agent()

        # Initialize reward models
        self.reward_models = self._init_reward_models()

    def _init_rl_agent(self):
        """Initialize the RL agent using hyperparameters from ExperimentManager."""
        wrapped_env = RewardVecEnvWrapper(
            self.env, 
            reward_fn=self.compute_ensemble_reward
        )

        if self.algorithm == "ppo":
            return PPO(
                env=wrapped_env, 
                verbose=1, 
                seed=self.seed,
                device=self.device,
                tensorboard_log=self.tensorboard_log,
                **self._hyperparams
            )
        else:
            return SAC(
                env=wrapped_env, 
                verbose=1, 
                seed=self.seed,
                device=self.device,
                tensorboard_log=self.tensorboard_log,
                **self._hyperparams
            )

    def _init_reward_models(self):
        """Initialize reward models for each feedback type."""
        reward_models = {}

        for feedback_type in self.feedback_types:
            if "ALE/" in self.env_name or "procgen" in self.env_name:
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
                    ensemble_count=self.num_ensemble_models,
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
                    ensemble_count=self.num_ensemble_models,
                )
            reward_models[feedback_type] = model

        return reward_models

    def collect_trajectories(self, n_trajectories: int) -> List[Dict]:
        """Collect trajectories using current policy."""
        trajectories = []
        initial_states = []

        for _ in range(n_trajectories):
            trajectory = []
            obs, _ = self.gen_env.reset()
            initial_states.append(self.gen_env.save_state(observation=obs))

            for _ in range(self.oracle.segment_len):
                action, _ = self.rl_agent.predict(obs, deterministic=False)
                next_obs, reward, terminated, truncated, _ = self.gen_env.step(action)
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
            action = vectorized_one_hot_vector(np.array(action), self.one_hot_dim)

        # add batch dimension to actions if not present (if called with non-vectorized env.)
        if len(action.shape) < 2:
            action = np.expand_dims(action, axis=0)
    
        # Convert to torch tensors of shape [batch_size, ...]
        state_tensor = torch.as_tensor(state, device=device, dtype=torch.float32).unsqueeze(1)
        action_tensor = torch.as_tensor(action, device=device, dtype=torch.float32).unsqueeze(1)
    
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
                    st_expanded = state_tensor.repeat(
                        reward_model.ensemble_count, *[1] * (len(state_tensor.shape) - 1)
                    )
                    act_expanded = action_tensor.repeat(
                        reward_model.ensemble_count, *[1] * (len(action_tensor.shape) - 1)
                    )
    
                    # predictions.shape might be [ensemble_count, batch_size]
                    # or [ensemble_count, batch_size, 1], etc.
                    predictions = reward_model(st_expanded, act_expanded)
    
                    # Make sure we reduce the final dimension if necessary
                    if predictions.dim() == 3 and predictions.shape[-1] == 1:
                        # e.g. shape: (ensemble_count, batch_size, 1)
                        predictions = predictions.squeeze(-1)
    
                    mean_reward, uncertainty = compute_grouped(predictions, reward_model.ensemble_count)
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
        if self.wandb_logger is not None:
            metrics = {"feedback_counts": feedback_counts, **reward_metrics}
            wandb.log(metrics)

        return feedback_counts, reward_metrics

    def train_rl_agent(self):
        """Train RL agent using current reward models."""
        # Use callbacks if provided
        callback = None if not self.callbacks else self.callbacks
        
        self.rl_agent.learn(
            total_timesteps=self.rl_steps_per_iteration,
            reset_num_timesteps=False,
            callback=callback,
            tb_log_name=self.tb_log_name,
        )

    def train(self, total_iterations: int, sampling_strategy: str = "random"):
        """Run full training loop for specified number of iterations."""
        # Initialize callbacks if they exist
        if self.callbacks:
            for callback in self.callbacks:
                callback.init_callback(self.rl_agent)

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

        if self.wandb_logger is not None:
            wandb.finish()

        # Cleanup callbacks if they exist
        if self.callbacks:
            for callback in self.callbacks:
                callback.on_training_end()

def main():
    parser = TrainingUtils.setup_base_parser()
    parser.add_argument(
        "--feedback-types",
        nargs="+",
        type=str,
        #default=["evaluative", "comparative", "demonstrative", "descriptive", "corrective", "descriptive_preference"],
        default=["evaluative", "comparative", "demonstrative", "corrective"],
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
        "--save-folder",
        type=str,
        default="trained_agents",
        help="Folder for finished feedback RL agents",
    )
    parser.add_argument(
        "--reference-data-folder",
        type=str,
        default="feedback",
        help="Folder containing pre-computed offline feedback for calibration",
    )
    parser.add_argument(
        "--n-feedback-per-iteration",
        type=int,
        default=30,
        help="Feedback Instances collected per iteration",
    )
    parser.add_argument(
        "--rl-steps-per-iteration",
        type=int,
        default=5000,
        help="Number of training iterations",
    )
    parser.add_argument(
        "--n-timesteps",
        type=int,
        default=-1,
        help="Overwrite for RL training timesteps",
    )
    parser.add_argument(
        "--reward-training-epochs",
        type=int,
        default=5,
        help="Number of epochs",
    )
    parser.add_argument(
        "--top-n-models", 
        type=int, 
        default=3, 
        help="Top N models to use"
    )
    parser.add_argument(
        "--expert-model-base-path", 
        type=str, 
        default="train_baselines/gt_agents", 
        help="Expert model base path"
    )
    parser.add_argument(
        "--feedback-buffer-size",
        type=int,
        default=1000,
        help="Maximum size of the feedback buffer",
    )
    parser.add_argument(
        "--num-ensemble-models",
        type=int,
        default=4,
        help="Number of ensemble models for masksemble",
    )
    args = parser.parse_args()

    # Setup oracle
    feedback_id, _ = TrainingUtils.get_model_ids(args)
    device = TrainingUtils.get_device()
    feedback_path = Path(args.reference_data_folder) / f"{feedback_id}.pkl"

    gen_environment = TrainingUtils.setup_environment(args.environment, args.seed)
    expert_models = TrainingUtils.load_expert_models(
        env_name=args.environment,
        algorithm=args.algorithm,
        checkpoints_path=str(get_project_root() / args.expert_model_base_path),
        environment=gen_environment,
        top_n_models=args.top_n_models,
    )

    oracle = FeedbackOracle(
        expert_models=expert_models,
        environment=gen_environment,
        reference_data_path=feedback_path,
        noise_level=args.noise_level,
    )
    unique_id = str(uuid.uuid4())[:8]
    
    #try:
    wandb.init(
        name=f"DYNAMIC_RL_{args.algorithm}_{args.environment}_{','.join(args.feedback_types)}_{unique_id}",
        project=args.wandb_project_name,
        config={
            "algorithm": args.algorithm,
            "feedback_types": args.feedback_types,
            "n_feedback_per_iteration": args.n_feedback_per_iteration,
            "rl_steps_per_iteration": args.rl_steps_per_iteration,
            "reward_training_epochs": args.reward_training_epochs,
            "feedback_buffer_size": args.feedback_buffer_size,
        },
        sync_tensorboard=True,
    )

    wandb_logger = WandbLogger(
        #roject=wandb_project_name,
        #name=f"DYNAMIC_RL_{algorithm}_{env_name}_{','.join(feedback_types)}",
    )
    #except:
    #    print("Could not initialze W&B")
    #    wandb_logger = None

    # Create expert manager
    exp_manager = ExperimentManager(
        args=args,
        algo=args.algorithm,
        env_id=args.environment,
        log_folder=args.save_folder,
        eval_freq=5000,
        n_eval_episodes=5,
        use_wandb_callback=True,
        tensorboard_log="tb_logs",
    )

    # Setup experiment and get hyperparameters
    hyperparams = exp_manager.get_hyperparam_config_for_algo()

    # Create environment
    rl_env = exp_manager.create_envs(n_envs=exp_manager.n_envs)

    # Create DynamicRLHF with ExperimentManager's hyperparameters and callbacks
    drlhf = DynamicRLHF(
        oracle=oracle,
        env=rl_env,
        gen_env=gen_environment, # normal gym env for creating trajectories
        env_name=args.environment,
        algorithm=args.algorithm,
        feedback_types=args.feedback_types,
        n_feedback_per_iteration=args.n_feedback_per_iteration,
        feedback_buffer_size=args.feedback_buffer_size,
        rl_steps_per_iteration=args.rl_steps_per_iteration,
        reward_training_epochs=args.reward_training_epochs,
        num_ensemble_models=args.num_ensemble_models,
        hyperparams=hyperparams,
        callbacks=exp_manager.callbacks,
        device=device,
        wandb_logger=wandb_logger,
        tensorboard_log=exp_manager.tensorboard_log,
        seed=args.seed,
    )

    # Train

    n_timesteps = args.n_timesteps if args.n_timesteps > 0 else exp_manager.n_timesteps
    total_iterations = max(1, n_timesteps // drlhf.rl_steps_per_iteration)
    drlhf.train(total_iterations=total_iterations, sampling_strategy="random")

if __name__ == "__main__":
    main()