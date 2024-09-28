import numpy as np
import torch
import os
import re
from typing import List, Tuple, Dict, Any, Union
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv
from sklearn.cluster import MiniBatchKMeans
from scipy.spatial.distance import cdist

class FeedbackOracle:
    def __init__(self, 
                 expert_models: List[Tuple[Union[PPO, SAC], VecNormalize]], 
                 environment,
                 checkpoints_path: str,
                 algorithm: str,
                 n_clusters: int = 100,
                 n_trajectories_per_checkpoint: int = 10,
                 segment_len: int = 50,
                 gamma: float = 0.99):
        self.expert_models = expert_models
        self.environment = environment
        self.checkpoints_path = checkpoints_path
        self.algorithm = algorithm
        self.n_clusters = n_clusters
        self.n_trajectories_per_checkpoint = n_trajectories_per_checkpoint
        self.segment_len = segment_len
        self.gamma = gamma
        
        # Pre-compute binning and clustering
        self.initialize_binning_and_clustering()

    def initialize_binning_and_clustering(self):
        # Generate trajectories from checkpoints
        trajectories = self.generate_trajectories_from_checkpoints()
        
        # Compute optimality gaps for binning
        self.opt_gaps = self.compute_optimality_gaps(trajectories)
        
        # Compute tolerance for preference comparisons
        self.tolerance = np.std(self.opt_gaps) / 10.0

        # Prepare data for clustering
        states_actions = np.concatenate([
            np.array([np.concatenate((step[0].squeeze(0), step[1])) for step in traj])
            for traj in trajectories
        ])
        
        rewards = np.concatenate([
            np.array([step[2] for step in traj])
            for traj in trajectories
        ])

        # Clustering
        batch_size = min(1000, len(states_actions) // 100)
        self.kmeans = MiniBatchKMeans(n_clusters=self.n_clusters, batch_size=batch_size, random_state=42)
        self.kmeans.fit(states_actions)
        self.cluster_representatives = self.kmeans.cluster_centers_
        
        # Compute average rewards per cluster for descriptive feedback
        cluster_assignments = self.kmeans.predict(states_actions)
        self.cluster_rewards = []
        for i in range(self.n_clusters):
            cluster_mask = cluster_assignments == i
            cluster_rewards = rewards[cluster_mask]
            if len(cluster_rewards) > 0:
                avg_reward = np.mean(cluster_rewards)
            else:
                avg_reward = 0.0
            self.cluster_rewards.append(avg_reward)
        
        # Compute ratings bins
        max_rating = 10
        self.ratings_bins = np.linspace(min(self.opt_gaps), max(self.opt_gaps), max_rating + 1)

    def generate_trajectories_from_checkpoints(self) -> List[List[Tuple[np.ndarray, np.ndarray, float, bool]]]:
        trajectories = []
        checkpoints_dir = os.path.join(self.checkpoints_path, self.algorithm, f"{self.environment.spec.id.replace('/', '-')}_1")
        checkpoint_files = [
            file for file in os.listdir(checkpoints_dir) if re.search(r"rl_model_.*\.zip", file)
        ]
        checkpoint_files = ["random"] + sorted(checkpoint_files, key=lambda x: int(re.search(r'\d+', x).group()) if x != "random" else -1)

        for model_file in checkpoint_files:
            if model_file != "random":
                model = (PPO if self.algorithm == "ppo" else SAC).load(
                    os.path.join(checkpoints_dir, model_file),
                    custom_objects={"learning_rate": 0.0, "lr_schedule": lambda _: 0.0},
                )
                norm_env_path = os.path.join(checkpoints_dir, self.environment.spec.id, "vecnormalize.pkl")
                norm_env = VecNormalize.load(norm_env_path, DummyVecEnv([lambda: self.environment])) if os.path.isfile(norm_env_path) else None
            else:
                model = None
                norm_env = None

            for _ in range(self.n_trajectories_per_checkpoint):
                trajectory = self.generate_single_trajectory(model, norm_env)
                trajectories.append(trajectory)

        return trajectories

    def generate_single_trajectory(self, model: Union[PPO, SAC, None], norm_env: VecNormalize) -> List[Tuple[np.ndarray, np.ndarray, float, bool]]:
        obs, _ = self.environment.reset()
        trajectory = []
        done = False
        while not done:
            if model is not None:
                action, _ = model.predict(norm_env.normalize_obs(obs) if norm_env else obs, deterministic=True)
            else:
                action = self.environment.action_space.sample()
            next_obs, reward, terminated, truncated, _ = self.environment.step(action)
            done = terminated or truncated
            trajectory.append((np.expand_dims(obs, axis=0), action, reward, done))
            obs = next_obs
        return trajectory

    def compute_optimality_gaps(self, trajectories: List[List[Tuple[np.ndarray, np.ndarray, float, bool]]]) -> List[float]:
        opt_gaps = []
        for traj in trajectories:
            initial_vals = [self.predict_expert_value(expert_model, traj[0][0]) for expert_model in self.expert_models]
            initial_val = np.mean(initial_vals)
            rewards = np.array([step[2] for step in traj])
            discounted_rew_sum = np.sum(rewards * (self.gamma ** np.arange(len(rewards))))
            final_vals = [self.predict_expert_value(expert_model, traj[-1][0]) for expert_model in self.expert_models]
            final_val = np.mean(final_vals)
            opt_gap = (initial_val - self.gamma ** len(rewards) * final_val) - discounted_rew_sum
            opt_gaps.append(opt_gap)
        return opt_gaps

    def predict_expert_value(self, expert_model: Tuple[Union[PPO, SAC], VecNormalize], observation: np.ndarray) -> float:
        model, norm_env = expert_model
        if norm_env is not None:
            observation = norm_env.normalize_obs(observation)
        observation = model.policy.obs_to_tensor(observation)[0]
        with torch.no_grad():
            if isinstance(model, SAC):
                actions = torch.zeros((observation.shape[0], model.action_space.shape[0]), device=observation.device)
                q_values = torch.cat(model.policy.critic(observation, actions), dim=1)
                value = q_values.mean(dim=1)
            else:
                value = model.policy.predict_values(observation)
            return value.item()

    def get_feedback(self, trajectory: List[Tuple[np.ndarray, np.ndarray, float, bool]], initial_state: np.ndarray, feedback_types: List[str]) -> Dict[str, Any]:
        feedback = {}
        
        if 'evaluative' in feedback_types:
            feedback['evaluative'] = self.get_evaluative_feedback(trajectory)
        
        if 'comparative' in feedback_types:
            # For comparative feedback, need another trajectory to compare
            # For demonstration purposes, we can select a random trajectory from precomputed ones
            other_trajectory = self.get_random_trajectory()
            feedback['comparative'] = self.get_comparative_feedback(trajectory, other_trajectory)
        
        if 'demonstrative' in feedback_types:
            feedback['demonstrative'] = self.get_demonstrative_feedback(initial_state)
        
        if 'corrective' in feedback_types:
            feedback['corrective'] = self.get_corrective_feedback(trajectory, initial_state)
        
        if 'descriptive' in feedback_types:
            feedback['descriptive'] = self.get_descriptive_feedback(trajectory)
        
        return feedback

    def get_evaluative_feedback(self, trajectory: List[Tuple[np.ndarray, np.ndarray, float, bool]]) -> int:
        opt_gap = self.compute_optimality_gaps([trajectory])[0]
        bin_index = np.digitize(opt_gap, self.ratings_bins) - 1  # Adjust index since np.digitize starts from 1
        rating = 10 - bin_index  # Ratings from 0 to 10, higher is better
        rating = max(0, min(10, rating))  # Ensure rating is within [0, 10]
        return int(rating)

    def get_comparative_feedback(self, trajectory1: List[Tuple[np.ndarray, np.ndarray, float, bool]], trajectory2: List[Tuple[np.ndarray, np.ndarray, float, bool]]) -> Tuple[int, int, int]:
        opt_gap1 = self.compute_optimality_gaps([trajectory1])[0]
        opt_gap2 = self.compute_optimality_gaps([trajectory2])[0]
        gap_diff = opt_gap1 - opt_gap2
        if abs(gap_diff) < self.tolerance:
            # Indifferent preference
            return (0, 1, 0)
        elif gap_diff > 0:
            # trajectory2 preferred over trajectory1
            return (1, 0, 1)
        else:
            # trajectory1 preferred over trajectory2
            return (0, 1, 1)

    def get_demonstrative_feedback(self, initial_state: np.ndarray) -> List[Tuple[np.ndarray, np.ndarray, float, bool, int]]:
        self.environment.reset()
        self.environment.load_state(initial_state)
        
        best_demo = None
        best_return = float('-inf')
        
        for exp_model_index, (expert_model, exp_norm_env) in enumerate(self.expert_models):
            demo = []
            obs = initial_state
            done = False
            for _ in range(self.segment_len):
                action, _ = expert_model.predict(exp_norm_env.normalize_obs(obs) if exp_norm_env else obs, deterministic=True)
                next_obs, reward, terminated, truncated, _ = self.environment.step(action)
                done = terminated or truncated
                demo.append((np.expand_dims(obs, axis=0), action, reward, done, exp_model_index))
                obs = next_obs
                if done:
                    break
            
            demo_return = sum(step[2] for step in demo)
            if demo_return > best_return:
                best_return = demo_return
                best_demo = demo
        
        return best_demo

    def get_corrective_feedback(self, trajectory: List[Tuple[np.ndarray, np.ndarray, float, bool]], initial_state: np.ndarray) -> Tuple[List[Tuple[np.ndarray, np.ndarray, float, bool]], Union[List[Tuple[np.ndarray, np.ndarray, float, bool, int]], None]]:
        expert_trajectory = self.get_demonstrative_feedback(initial_state)
        
        if sum(step[2] for step in expert_trajectory) > sum(step[2] for step in trajectory):
            return (trajectory, expert_trajectory)
        else:
            return (trajectory, None)

    def get_descriptive_feedback(self, trajectory: List[Tuple[np.ndarray, np.ndarray, float, bool]]) -> Tuple[np.ndarray, np.ndarray, float]:
        states_actions = np.array([np.concatenate((step[0].squeeze(0), step[1])) for step in trajectory])
        avg_state_action = np.mean(states_actions, axis=0)
        
        distances = cdist([avg_state_action], self.cluster_representatives)
        most_similar_cluster = np.argmin(distances)
        cluster_representative = self.cluster_representatives[most_similar_cluster]
        cluster_reward = self.cluster_rewards[most_similar_cluster]
        
        obs_dim = trajectory[0][0].squeeze(0).shape[0]
        return (
            cluster_representative[:obs_dim],  # state
            cluster_representative[obs_dim:],  # action
            cluster_reward  # average reward of the cluster
        )

    def get_random_trajectory(self) -> List[Tuple[np.ndarray, np.ndarray, float, bool]]:
        # Generate a random trajectory from the environment
        obs, _ = self.environment.reset()
        trajectory = []
        done = False
        while not done and len(trajectory) < self.segment_len:
            action = self.environment.action_space.sample()
            next_obs, reward, terminated, truncated, _ = self.environment.step(action)
            done = terminated or truncated
            trajectory.append((np.expand_dims(obs, axis=0), action, reward, done))
            obs = next_obs
        return trajectory
