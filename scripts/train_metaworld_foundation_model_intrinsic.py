#!/usr/bin/env python3
"""
Intrinsically motivated pre-training for Metaworld foundation model.
Based on PEBBLE's unsupervised exploration approach using state entropy maximization
across multiple Metaworld tasks to collect diverse states for foundation model training.
"""

import argparse
import time
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
import random
import pickle as pkl
import gymnasium as gym
from typing import List, Optional
from metaworld.policies import ENV_POLICY_MAP

# Stable Baselines3 imports
from stable_baselines3 import SAC

# Local imports
from multi_type_feedback.utils import TrainingUtils
from multi_type_feedback.networks import SingleNetwork
from multi_type_feedback.wandb_logger import WandbLogger


class MetaworldMultiTaskEnv:
    """Multi-task Metaworld environment wrapper for foundation model training."""
    
    def __init__(self, task_names: List[str], seed: int = 0):
        self.task_names = task_names
        self.current_task_idx = 0
        self.seed = seed
        self.envs = {}
        self._setup_environments()
        
    def _setup_environments(self):
        """Initialize all Metaworld environments."""
        for task_name in self.task_names:
            env_name = f"metaworld-{task_name}"
            env = TrainingUtils.setup_environment(env_name, seed=self.seed, save_reset_wrapper=False)
            self.envs[task_name] = env
            
    def get_current_env(self) -> gym.Env:
        """Get the current active environment."""
        task_name = self.task_names[self.current_task_idx]
        return self.envs[task_name]
    
    def switch_task(self, task_idx: Optional[int] = None):
        """Switch to a different task."""
        if task_idx is None:
            self.current_task_idx = (self.current_task_idx + 1) % len(self.task_names)
        else:
            self.current_task_idx = task_idx % len(self.task_names)
    
    def get_random_task(self) -> str:
        """Get a random task name."""
        return random.choice(self.task_names)
    
    def reset(self, task_name: Optional[str] = None):
        """Reset environment, optionally switching to a specific task."""
        if task_name:
            self.current_task_idx = self.task_names.index(task_name)
        return self.get_current_env().reset()
    
    def step(self, action):
        """Step the current environment."""
        return self.get_current_env().step(action)
    
    @property
    def observation_space(self):
        return self.get_current_env().observation_space
    
    @property
    def action_space(self):
        return self.get_current_env().action_space


class StateEntropyReplayBuffer:
    """Replay buffer for intrinsic motivation with state entropy calculation."""
    
    def __init__(self, obs_shape, action_shape, capacity, device, window=1):
        self.capacity = capacity
        self.device = device
        self.window = window
        
        # Storage arrays
        obs_dtype = np.float32 if len(obs_shape) == 1 else np.uint8
        self.obses = np.empty((capacity, *obs_shape), dtype=obs_dtype)
        self.next_obses = np.empty((capacity, *obs_shape), dtype=obs_dtype)
        self.actions = np.empty((capacity, *action_shape), dtype=np.float32)
        self.intrinsic_rewards = np.empty((capacity, 1), dtype=np.float32)
        self.not_dones = np.empty((capacity, 1), dtype=np.float32)
        self.not_dones_no_max = np.empty((capacity, 1), dtype=np.float32)
        self.task_ids = np.empty((capacity, 1), dtype=np.int32)
        
        self.idx = 0
        self.full = False
        
    def __len__(self):
        return self.capacity if self.full else self.idx
    
    def add(self, obs, action, int_reward, next_obs, done, done_no_max, task_id):
        """Add a transition to the buffer."""
        np.copyto(self.obses[self.idx], obs)
        np.copyto(self.actions[self.idx], action)
        np.copyto(self.intrinsic_rewards[self.idx], int_reward)
        np.copyto(self.next_obses[self.idx], next_obs)
        np.copyto(self.not_dones[self.idx], not done)
        np.copyto(self.not_dones_no_max[self.idx], not done_no_max)
        np.copyto(self.task_ids[self.idx], task_id)
        
        self.idx = (self.idx + 1) % self.capacity
        self.full = self.full or self.idx == 0
    
    def sample(self, batch_size):
        """Sample a batch of transitions."""
        idxs = np.random.randint(0, self.capacity if self.full else self.idx, size=batch_size)
        
        obses = torch.as_tensor(self.obses[idxs], device=self.device).float()
        actions = torch.as_tensor(self.actions[idxs], device=self.device)
        int_rewards = torch.as_tensor(self.intrinsic_rewards[idxs], device=self.device)
        next_obses = torch.as_tensor(self.next_obses[idxs], device=self.device).float()
        not_dones = torch.as_tensor(self.not_dones[idxs], device=self.device)
        not_dones_no_max = torch.as_tensor(self.not_dones_no_max[idxs], device=self.device)
        task_ids = torch.as_tensor(self.task_ids[idxs], device=self.device)
        
        return obses, actions, int_rewards, next_obses, not_dones, not_dones_no_max, task_ids
    
    def sample_full_obs(self, batch_size=512):
        """Sample observations for state entropy calculation."""
        if self.full:
            full_obs = self.obses
        else:
            full_obs = self.obses[:self.idx]
        
        if len(full_obs) < batch_size:
            batch_size = len(full_obs)
            
        full_idxs = np.random.choice(full_obs.shape[0], size=batch_size, replace=False)
        full_obs = torch.as_tensor(full_obs[full_idxs], device=self.device)
        return full_obs


class StateEntropyAgent:
    """Agent for intrinsic motivation based on state entropy maximization."""
    
    def __init__(self, obs_dim, action_dim, hidden_dim=256, lr=3e-4, device='cuda'):
        self.device = device
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        
        # State representation network
        self.state_encoder = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2)
        ).to(device)
        
        # Actor-critic networks (using SAC backbone)
        self.actor_critic = SAC(
            'MlpPolicy',
            env=None,  # Will be set later
            learning_rate=lr,
            buffer_size=100000,
            learning_starts=1000,
            batch_size=256,
            tau=0.005,
            gamma=0.99,
            device=device,
            verbose=0
        )
        
        self.state_optimizer = torch.optim.Adam(self.state_encoder.parameters(), lr=lr)
        
    def get_state_representation(self, obs):
        """Get state representation for entropy calculation."""
        return self.state_encoder(obs)
    
    def compute_intrinsic_reward(self, obs, full_obs, k=10):
        """Compute intrinsic reward based on state entropy (k-NN based)."""
        # Get state representations
        state_rep = self.get_state_representation(obs)
        full_state_reps = self.get_state_representation(full_obs)
        
        # Compute k-NN distances
        dists = torch.cdist(state_rep, full_state_reps, p=2)
        knn_dists, _ = torch.topk(dists, k=k, largest=False, dim=1)
        
        # Intrinsic reward is based on distance to k-th nearest neighbor
        intrinsic_reward = knn_dists[:, -1].unsqueeze(1)  # Distance to k-th neighbor
        
        return intrinsic_reward
    
    def update_state_encoder(self, obs_batch, target_diversity_loss):
        """Update state encoder to maximize state diversity."""
        self.state_optimizer.zero_grad()
        
        state_reps = self.get_state_representation(obs_batch)
        
        # Maximize pairwise distances (diversity loss)
        pairwise_dists = torch.cdist(state_reps, state_reps, p=2)
        diversity_loss = -torch.mean(pairwise_dists)  # Negative because we want to maximize
        
        diversity_loss.backward()
        self.state_optimizer.step()
        
        return diversity_loss.item()


class IntrinsicMotivationWorkspace:
    """Main workspace for intrinsic motivation pre-training."""
    
    def __init__(self, args):
        self.args = args
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Setup seeds
        TrainingUtils.set_seeds(args.seed)
        
        # Get metaworld tasks from ENV_POLICY_MAP (automatically includes v3 tasks)
        self.metaworld_tasks = list(ENV_POLICY_MAP.keys())
        
        # Use subset of tasks if specified
        if args.n_tasks > 0:
            self.metaworld_tasks = self.metaworld_tasks[:args.n_tasks]
        
        # Setup multi-task environment
        self.multi_env = MetaworldMultiTaskEnv(self.metaworld_tasks, seed=args.seed)
        
        # Get environment specs
        obs_dim = self.multi_env.observation_space.shape[0]
        action_dim = self.multi_env.action_space.shape[0]
        
        # Setup replay buffer
        self.replay_buffer = StateEntropyReplayBuffer(
            self.multi_env.observation_space.shape,
            self.multi_env.action_space.shape,
            args.buffer_capacity,
            self.device
        )
        
        # Setup agent
        self.agent = StateEntropyAgent(
            obs_dim, action_dim, 
            hidden_dim=args.hidden_dim,
            lr=args.learning_rate,
            device=self.device
        )
        
        # Setup foundation model (reward model architecture)
        input_spaces = (self.multi_env.observation_space, self.multi_env.action_space)
        self.foundation_model = SingleNetwork(
            input_spaces=input_spaces,
            layer_num=args.foundation_layers,
            output_dim=1,  # Single output for general reward prediction
            hidden_dim=args.foundation_hidden_dim,
            action_hidden_dim=args.foundation_hidden_dim,
            loss_function=lambda net, batch: nn.MSELoss()(net(*batch[0]), batch[1]),
            learning_rate=args.foundation_lr
        ).to(self.device)
        
        # Setup logging
        self.logger = WandbLogger(
            project_name=args.wandb_project,
            run_name=f"metaworld_foundation_intrinsic_{args.seed}",
            config=vars(args)
        )
        
        # Training state
        self.step = 0
        self.episode = 0
        self.task_switch_frequency = args.task_switch_frequency
        
    def evaluate_diversity(self, n_episodes=5):
        """Evaluate state diversity across tasks."""
        all_states = []
        
        for task_name in self.metaworld_tasks[:min(10, len(self.metaworld_tasks))]:  # Evaluate on subset
            obs = self.multi_env.reset(task_name)
            episode_states = [obs[0] if isinstance(obs, tuple) else obs]
            
            for _ in range(100):  # Short episodes for evaluation
                action = self.multi_env.action_space.sample()
                obs, _, done, _ = self.multi_env.step(action)
                episode_states.append(obs)
                
                if done:
                    break
                    
            all_states.extend(episode_states)
        
        # Compute state diversity metrics
        if len(all_states) > 1:
            states_tensor = torch.tensor(np.array(all_states), device=self.device, dtype=torch.float32)
            state_reps = self.agent.get_state_representation(states_tensor)
            
            # Compute average pairwise distance
            pairwise_dists = torch.cdist(state_reps, state_reps, p=2)
            avg_pairwise_dist = torch.mean(pairwise_dists).item()
            
            # Compute state coverage (standard deviation)
            state_std = torch.std(state_reps, dim=0).mean().item()
            
            return {
                'avg_pairwise_distance': avg_pairwise_dist,
                'state_coverage': state_std,
                'n_states_collected': len(all_states)
            }
        
        return {'avg_pairwise_distance': 0.0, 'state_coverage': 0.0, 'n_states_collected': 0}
    
    def run_exploration(self):
        """Run intrinsic motivation exploration phase."""
        print("Starting intrinsic motivation exploration...")
        
        # Initialize environment
        current_task = self.multi_env.get_random_task()
        obs = self.multi_env.reset(current_task)
        obs = obs[0] if isinstance(obs, tuple) else obs
        
        episode_reward = 0
        episode_intrinsic_reward = 0
        episode_step = 0
        
        start_time = time.time()
        
        while self.step < self.args.exploration_steps:
            # Switch tasks periodically
            if self.step % self.task_switch_frequency == 0 and self.step > 0:
                current_task = self.multi_env.get_random_task()
                obs = self.multi_env.reset(current_task)
                obs = obs[0] if isinstance(obs, tuple) else obs
                episode_step = 0
            
            # Sample action (random during exploration)
            action = self.multi_env.action_space.sample()
            
            # Environment step
            next_obs, _, done, info = self.multi_env.step(action)
            
            # Compute intrinsic reward
            obs_tensor = torch.tensor(obs, device=self.device, dtype=torch.float32).unsqueeze(0)
            if len(self.replay_buffer) > 100:  # Need some states for comparison
                full_obs = self.replay_buffer.sample_full_obs(min(512, len(self.replay_buffer)))
                int_reward = self.agent.compute_intrinsic_reward(obs_tensor, full_obs, k=min(10, len(self.replay_buffer)//10))
                int_reward = int_reward.cpu().numpy()[0, 0]
            else:
                int_reward = 1.0  # High reward for early exploration
            
            # Store in replay buffer
            task_id = self.metaworld_tasks.index(current_task)
            done_no_max = False if episode_step == 500 else done  # Metaworld max episode length
            
            self.replay_buffer.add(
                obs, action, int_reward, next_obs, 
                done, done_no_max, task_id
            )
            
            # Update state encoder periodically
            if self.step % self.args.state_update_freq == 0 and len(self.replay_buffer) > self.args.batch_size:
                obs_batch, _, _, _, _, _, _ = self.replay_buffer.sample(self.args.batch_size)
                diversity_loss = self.agent.update_state_encoder(obs_batch, None)
                
                if self.step % 1000 == 0:
                    self.logger.log({'exploration/diversity_loss': diversity_loss}, self.step)
            
            # Update counters
            episode_reward += 0  # No extrinsic reward used
            episode_intrinsic_reward += int_reward
            episode_step += 1
            self.step += 1
            
            # Handle episode end
            if done or episode_step >= 500:
                self.logger.log({
                    'exploration/episode_reward': episode_reward,
                    'exploration/episode_intrinsic_reward': episode_intrinsic_reward,
                    'exploration/episode_length': episode_step,
                    'exploration/current_task': task_id
                }, self.step)
                
                # Reset for new episode
                current_task = self.multi_env.get_random_task()
                obs = self.multi_env.reset(current_task)
                obs = obs[0] if isinstance(obs, tuple) else obs
                episode_reward = 0
                episode_intrinsic_reward = 0
                episode_step = 0
                self.episode += 1
            else:
                obs = next_obs
            
            # Periodic evaluation
            if self.step % self.args.eval_freq == 0:
                diversity_metrics = self.evaluate_diversity()
                self.logger.log({
                    'exploration/avg_pairwise_distance': diversity_metrics['avg_pairwise_distance'],
                    'exploration/state_coverage': diversity_metrics['state_coverage'],
                    'exploration/buffer_size': len(self.replay_buffer)
                }, self.step)
                
                print(f"Step {self.step}: Diversity metrics: {diversity_metrics}")
        
        print(f"Exploration completed! Collected {len(self.replay_buffer)} transitions.")
    
    def train_foundation_model(self):
        """Train foundation model on collected diverse data."""
        print("Training foundation model...")
        
        # Prepare dataset from replay buffer
        if len(self.replay_buffer) < 1000:
            print("Not enough data for foundation model training!")
            return
        
        # Sample diverse training data
        n_samples = min(len(self.replay_buffer), self.args.foundation_training_samples)
        obs, actions, int_rewards, next_obs, _, _, task_ids = self.replay_buffer.sample(n_samples)
        
        # Prepare training data (obs, actions) -> intrinsic reward only
        combined_rewards = int_rewards
        
        # Convert to sequence format expected by SingleNetwork
        batch_size = obs.shape[0]
        obs_seq = obs.unsqueeze(1)  # Add sequence dimension
        actions_seq = actions.unsqueeze(1)
        rewards_seq = combined_rewards.squeeze(1)
        
        print(f"Training foundation model on {n_samples} samples...")
        
        # Training loop
        optimizer = torch.optim.Adam(self.foundation_model.parameters(), lr=self.args.foundation_lr)
        
        for epoch in range(self.args.foundation_epochs):
            # Shuffle data
            perm = torch.randperm(batch_size)
            obs_shuffled = obs_seq[perm]
            actions_shuffled = actions_seq[perm]
            rewards_shuffled = rewards_seq[perm]
            
            total_loss = 0
            n_batches = 0
            
            # Mini-batch training
            for i in range(0, batch_size, self.args.foundation_batch_size):
                end_i = min(i + self.args.foundation_batch_size, batch_size)
                
                obs_batch = obs_shuffled[i:end_i]
                actions_batch = actions_shuffled[i:end_i]
                rewards_batch = rewards_shuffled[i:end_i]
                
                # Forward pass
                pred_rewards = self.foundation_model(obs_batch, actions_batch).squeeze(-1).squeeze(-1)
                
                # Compute loss
                loss = nn.MSELoss()(pred_rewards, rewards_batch)
                
                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                n_batches += 1
            
            avg_loss = total_loss / n_batches
            
            if epoch % 10 == 0:
                print(f"Foundation model epoch {epoch}: loss = {avg_loss:.4f}")
                self.logger.log({'foundation/training_loss': avg_loss}, epoch)
        
        print("Foundation model training completed!")
    
    def save_models(self):
        """Save trained models."""
        save_dir = Path(self.args.save_dir) / f"metaworld_foundation_intrinsic_{self.args.seed}"
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # Save foundation model
        torch.save(self.foundation_model.state_dict(), save_dir / "foundation_model.pth")
        
        # Save state encoder
        torch.save(self.agent.state_encoder.state_dict(), save_dir / "state_encoder.pth")
        
        # Save replay buffer
        buffer_data = {
            'obses': self.replay_buffer.obses[:len(self.replay_buffer)],
            'actions': self.replay_buffer.actions[:len(self.replay_buffer)],
            'intrinsic_rewards': self.replay_buffer.intrinsic_rewards[:len(self.replay_buffer)],
            'task_ids': self.replay_buffer.task_ids[:len(self.replay_buffer)]
        }
        
        with open(save_dir / "replay_buffer.pkl", 'wb') as f:
            pkl.dump(buffer_data, f)
        
        print(f"Models saved to {save_dir}")
    
    def run(self):
        """Main training loop."""
        print("Starting Metaworld foundation model training with intrinsic motivation...")
        print(f"Using {len(self.metaworld_tasks)} Metaworld tasks")
        
        # Phase 1: Exploration with intrinsic motivation
        self.run_exploration()
        
        # Phase 2: Train foundation model
        self.train_foundation_model()
        
        # Save models
        self.save_models()
        
        print("Training completed successfully!")


def main():
    parser = argparse.ArgumentParser(description="Train Metaworld foundation model with intrinsic motivation")
    
    # Environment settings
    parser.add_argument('--n-tasks', type=int, default=len(ENV_POLICY_MAP), help='Number of Metaworld tasks to use')
    parser.add_argument('--task-switch-frequency', type=int, default=1000, help='Steps between task switches')
    
    # Exploration settings
    parser.add_argument('--exploration-steps', type=int, default=1000000, help='Number of exploration steps')
    parser.add_argument('--buffer-capacity', type=int, default=1000000, help='Replay buffer capacity')
    parser.add_argument('--state-update-freq', type=int, default=100, help='State encoder update frequency')
    parser.add_argument('--eval-freq', type=int, default=10000, help='Evaluation frequency')
    
    # Agent settings
    parser.add_argument('--hidden-dim', type=int, default=256, help='Hidden dimension for state encoder')
    parser.add_argument('--learning-rate', type=float, default=3e-4, help='Learning rate for agent')
    parser.add_argument('--batch-size', type=int, default=256, help='Batch size for updates')
    
    # Foundation model settings
    parser.add_argument('--foundation-layers', type=int, default=3, help='Number of layers in foundation model')
    parser.add_argument('--foundation-hidden-dim', type=int, default=256, help='Hidden dimension for foundation model')
    parser.add_argument('--foundation-lr', type=float, default=1e-3, help='Learning rate for foundation model')
    parser.add_argument('--foundation-epochs', type=int, default=100, help='Training epochs for foundation model')
    parser.add_argument('--foundation-batch-size', type=int, default=128, help='Batch size for foundation model')
    parser.add_argument('--foundation-training-samples', type=int, default=100000, help='Number of samples for foundation training')
    parser.add_argument('--intrinsic-weight', type=float, default=0.1, help='Weight for intrinsic rewards in foundation model')
    
    # General settings
    parser.add_argument('--seed', type=int, default=0, help='Random seed')
    parser.add_argument('--save-dir', type=str, default='./models', help='Directory to save models')
    parser.add_argument('--wandb-project', type=str, default='metaworld-foundation-intrinsic', help='W&B project name')
    
    args = parser.parse_args()
    
    # Create workspace and run
    workspace = IntrinsicMotivationWorkspace(args)
    workspace.run()


if __name__ == '__main__':
    main()