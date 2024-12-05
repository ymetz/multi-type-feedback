from typing import Any, List, Tuple, Union

import numpy as np
import pickle
from scipy.spatial.distance import cdist
from sklearn.cluster import MiniBatchKMeans
from stable_baselines3.common.vec_env import VecNormalize
from stable_baselines3 import PPO, SAC


class FeedbackOracle:
    """Previous initialization and helper methods remain the same..."""

    def __init__(
        self,
        expert_models: List[Tuple[Union[PPO, SAC], VecNormalize]],
        environment,
        reference_data_path: str,  # Path to pre-computed reference data
        segment_len: int = 50,
        gamma: float = 0.99,
        noise_level: float = 0.0,
        n_clusters: int = 100,
    ):
        self.expert_models = expert_models
        self.environment = environment
        self.segment_len = segment_len
        self.gamma = gamma
        self.noise_level = noise_level
        self.n_clusters = n_clusters

        # Load and process reference data for calibration
        self.initialize_calibration(reference_data_path)

    def initialize_calibration(self, reference_data_path: str):
        """Initialize calibration data from pre-computed reference trajectories."""
        with open(reference_data_path, "rb") as f:
            reference_data = pickle.load(f)

        # Store reference optimality gaps for evaluative feedback calibration
        self.reference_opt_gaps = reference_data["opt_gaps"]
        max_rating = 10
        self.ratings_bins = np.linspace(
            min(self.reference_opt_gaps), max(self.reference_opt_gaps), max_rating + 1
        )

        # Process state-action pairs for clustering (descriptive feedback)
        states_actions = []
        for segment in reference_data["segments"]:
            states_actions.extend(
                [np.concatenate((step[0].squeeze(0), step[1])) for step in segment]
            )
        states_actions = np.array(states_actions)

        # Fit clustering model
        batch_size = min(1000, len(states_actions) // 100)
        self.kmeans = MiniBatchKMeans(
            n_clusters=self.n_clusters, batch_size=batch_size, random_state=42
        )
        self.kmeans.fit(states_actions)

        # Store cluster representatives and their average returns
        self.cluster_representatives = []
        self.cluster_rewards = []

        # Calculate average rewards for each cluster
        cluster_assignments = self.kmeans.predict(states_actions)
        rewards = []
        for segment in reference_data["segments"]:
            rewards.extend([step[2] for step in segment])
        rewards = np.array(rewards)

        for i in range(self.n_clusters):
            cluster_mask = cluster_assignments == i
            if np.any(cluster_mask):
                center = self.kmeans.cluster_centers_[i]
                avg_reward = np.mean(rewards[cluster_mask])
                self.cluster_representatives.append(center)
                self.cluster_rewards.append(avg_reward)

        self.cluster_representatives = np.array(self.cluster_representatives)
        self.cluster_rewards = np.array(self.cluster_rewards)

    def get_feedback(
        self,
        trajectory_data: Union[
            List[Tuple[np.ndarray, np.ndarray, float, bool]],  # single trajectory
            Tuple[
                List[Tuple[np.ndarray, np.ndarray, float, bool]],
                List[Tuple[np.ndarray, np.ndarray, float, bool]],
            ],  # trajectory pair
        ],
        initial_state: np.ndarray,
        feedback_type: str,
    ) -> Any:
        """
        Get feedback of specified type for either a single trajectory or a pair of trajectories.

        Args:
            trajectory_data: Either a single trajectory or a tuple of two trajectories
            initial_state: Initial state of the trajectory for demonstrative/corrective feedback
            feedback_type: Type of feedback to provide

        Returns:
            Feedback of the specified type
        """
        if feedback_type in [
            "evaluative",
            "demonstrative",
            "corrective",
            "descriptive",
        ]:
            if not isinstance(trajectory_data, list):
                raise ValueError(
                    f"{feedback_type} feedback requires a single trajectory"
                )

            if feedback_type == "evaluative":
                return self.get_evaluative_feedback(trajectory_data)
            elif feedback_type == "demonstrative":
                return self.get_demonstrative_feedback(initial_state)
            elif feedback_type == "corrective":
                return self.get_corrective_feedback(trajectory_data, initial_state)
            elif feedback_type == "descriptive":
                return self.get_descriptive_feedback(trajectory_data)

        elif feedback_type in ["comparative", "descriptive_preference"]:
            if not isinstance(trajectory_data, tuple) or len(trajectory_data) != 2:
                raise ValueError(
                    f"{feedback_type} feedback requires a pair of trajectories"
                )

            trajectory1, trajectory2 = trajectory_data

            if feedback_type == "comparative":
                return self.get_comparative_feedback(trajectory1, trajectory2)
            elif feedback_type == "descriptive_preference":
                return self.get_descriptive_preference_feedback(
                    trajectory1, trajectory2
                )

        else:
            raise ValueError(f"Unknown feedback type: {feedback_type}")

    def get_evaluative_feedback(
        self, trajectory: List[Tuple[np.ndarray, np.ndarray, float, bool]]
    ) -> int:
        """Return a calibrated rating between 0 and 10 based on optimality gap."""
        opt_gap = -self._compute_discounted_return(trajectory)  # Negative return as gap

        # Add noise if specified
        if self.noise_level > 0:
            opt_gap += np.random.normal(
                0, self.noise_level * np.std(self.reference_opt_gaps)
            )

        # Use pre-computed bins to determine rating
        bin_index = np.digitize(opt_gap, self.ratings_bins) - 1
        rating = 10 - bin_index  # Higher rating for lower optimality gap
        rating = max(0, min(10, rating))

        return int(rating)

    def get_descriptive_feedback(
        self, trajectory: List[Tuple[np.ndarray, np.ndarray, float, bool]]
    ) -> Tuple[np.ndarray, np.ndarray, float]:
        """Return the most similar cluster representative and its average reward."""
        # Compute average state-action for the trajectory
        states_actions = np.array(
            [np.concatenate((step[0].squeeze(0), step[1])) for step in trajectory]
        )
        avg_state_action = np.mean(states_actions, axis=0)

        # Find closest cluster
        distances = cdist([avg_state_action], self.cluster_representatives)
        closest_cluster = np.argmin(distances)

        # Get cluster representative
        representative = self.cluster_representatives[closest_cluster]
        reward = self.cluster_rewards[closest_cluster]

        # Add noise to reward if specified
        if self.noise_level > 0:
            reward += np.random.normal(
                0, self.noise_level * np.std(self.cluster_rewards)
            )

        # Split into state and action components
        obs_dim = trajectory[0][0].squeeze(0).shape[0]
        return (
            representative[:obs_dim],  # state
            representative[obs_dim:],  # action
            reward,
        )

    def get_comparative_feedback(self, trajectory1, trajectory2):
        """Existing implementation..."""
        return1 = self._compute_discounted_return(trajectory1)
        return2 = self._compute_discounted_return(trajectory2)

        if self.noise_level > 0:
            return1 += np.random.normal(0, self.noise_level * abs(return1))
            return2 += np.random.normal(0, self.noise_level * abs(return2))

        total_return = abs(return1) + abs(return2)
        if total_return == 0:
            return (0, 1, 0)

        diff = abs(return1 - return2) / total_return
        if diff < 0.1:
            return (0, 1, 0)
        elif return1 > return2:
            return (1, 0, 1)
        else:
            return (0, 1, 1)

    def get_demonstrative_feedback(self, initial_state):
        """Existing implementation..."""
        best_demo = None
        best_return = float("-inf")

        for exp_model_index, (expert_model, exp_norm_env) in enumerate(
            self.expert_models
        ):
            self.environment.reset()
            obs = initial_state.squeeze(0)
            demo = []

            for _ in range(self.segment_len):
                action, _ = expert_model.predict(
                    exp_norm_env.normalize_obs(obs) if exp_norm_env else obs,
                    deterministic=True,
                )
                next_obs, reward, terminated, truncated, _ = self.environment.step(
                    action
                )
                done = terminated or truncated
                demo.append(
                    (np.expand_dims(obs, axis=0), action, reward, done, exp_model_index)
                )
                if done:
                    break
                obs = next_obs

            demo_return = self._compute_discounted_return(demo)
            if demo_return > best_return:
                best_return = demo_return
                best_demo = demo

        return best_demo

    def get_corrective_feedback(self, trajectory, initial_state):
        """Existing implementation..."""
        trajectory_return = self._compute_discounted_return(trajectory)
        expert_demo = self.get_demonstrative_feedback(initial_state)
        expert_return = self._compute_discounted_return(expert_demo)

        if self.noise_level > 0:
            expert_return += np.random.normal(0, self.noise_level * abs(expert_return))
            trajectory_return += np.random.normal(
                0, self.noise_level * abs(trajectory_return)
            )

        if expert_return > trajectory_return:
            return (trajectory, expert_demo)
        return (trajectory, None)

    def get_random_trajectory(self):
        """Existing implementation..."""
        self.environment.reset()
        trajectory = []
        done = False

        while not done and len(trajectory) < self.segment_len:
            action = self.environment.action_space.sample()
            next_obs, reward, terminated, truncated, _ = self.environment.step(action)
            done = terminated or truncated
            trajectory.append((np.expand_dims(next_obs, axis=0), action, reward, done))

        return trajectory

    def _compute_discounted_return(self, trajectory: List[Tuple]) -> float:
        """Helper method to compute discounted return of a trajectory."""
        rewards = [step[2] for step in trajectory]
        return sum(reward * (self.gamma**i) for i, reward in enumerate(rewards))

    def get_descriptive_preference_feedback(
        self,
        trajectory1: List[Tuple[np.ndarray, np.ndarray, float, bool]],
        trajectory2: List[Tuple[np.ndarray, np.ndarray, float, bool]],
    ) -> Tuple[int, int, int]:
        """Compare two trajectories based on their closest cluster representatives."""
        # Get cluster information for both trajectories
        states_actions1 = np.array(
            [np.concatenate((step[0].squeeze(0), step[1])) for step in trajectory1]
        )
        states_actions2 = np.array(
            [np.concatenate((step[0].squeeze(0), step[1])) for step in trajectory2]
        )

        avg_state_action1 = np.mean(states_actions1, axis=0)
        avg_state_action2 = np.mean(states_actions2, axis=0)

        # Find closest clusters
        distances1 = cdist([avg_state_action1], self.cluster_representatives)
        distances2 = cdist([avg_state_action2], self.cluster_representatives)

        cluster1 = np.argmin(distances1)
        cluster2 = np.argmin(distances2)

        reward1 = self.cluster_rewards[cluster1]
        reward2 = self.cluster_rewards[cluster2]

        # Add noise if specified
        if self.noise_level > 0:
            noise_scale = self.noise_level * np.std(self.cluster_rewards)
            reward1 += np.random.normal(0, noise_scale)
            reward2 += np.random.normal(0, noise_scale)

        # Compare cluster rewards
        reward_diff = reward1 - reward2
        total_reward = abs(reward1) + abs(reward2)

        if total_reward == 0:
            return (0, 1, 0)  # Indifferent if both rewards are 0

        diff = abs(reward_diff) / total_reward

        # If difference is small, mark as indifferent
        if diff < 0.1:  # 10% threshold for indifference
            return (0, 1, 0)
        elif reward1 > reward2:
            return (1, 0, 1)
        else:
            return (0, 1, 1)
