import argparse
import bisect
import itertools
import os
import pickle
import random
import re
from collections import deque
from pathlib import Path
from typing import Iterator, List, Tuple, Type, Union

import gymnasium as gym
import numpy as np
import pandas as pd
import torch
from gymnasium.wrappers.stateful_observation import FrameStackObservation
from gymnasium.wrappers.transform_observation import TransformObservation
from minigrid.wrappers import FlatObsWrapper
from numpy.typing import NDArray
from procgen import ProcgenGym3Env
from rl_zoo3.wrappers import Gym3ToGymnasium
from sklearn.cluster import MiniBatchKMeans
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.atari_wrappers import WarpFrame
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from torch import Tensor

from rlhf.save_reset_wrapper import SaveResetEnvWrapper


def predict_expert_value(
    expert_model: Union[PPO, SAC], observation: np.ndarray, actions: Tensor = None
) -> Tensor:
    """Return the value from the expert's value function for a given observation and actions."""
    expert_model, norm_env = expert_model
    if norm_env is not None:
        observation = norm_env.normalize_obs(observation)

    observation = expert_model.policy.obs_to_tensor(observation)[0]
    with torch.no_grad():
        return torch.min(
            (
                torch.cat(
                    expert_model.policy.critic_target(observation, actions), dim=1
                )
                if isinstance(expert_model, SAC)
                else expert_model.policy.predict_values(observation)
            ),
            dim=1,
            keepdim=True,
        )[0]


class SegmentBuffer:
    """Memory-efficient buffer for storing segments using a sliding window."""

    def __init__(self, maxlen: int):
        self.buffer = deque(maxlen=maxlen)
        self.done_indices = []

    def append(self, item: tuple, step: int, done: bool):
        self.buffer.append(item)
        if done:
            self.done_indices.append(step)

    def get_segment(self, start: int, length: int) -> list:
        """Extract a segment avoiding done transitions."""
        end = start + length

        # Find relevant done indices
        insert_pos = bisect.bisect_left(self.done_indices, start)
        relevant_done_indices = []
        while (
            insert_pos < len(self.done_indices) and self.done_indices[insert_pos] < end
        ):
            relevant_done_indices.append(self.done_indices[insert_pos])
            insert_pos += 1

        if not relevant_done_indices:
            return list(itertools.islice(self.buffer, start, end))

        # Find longest valid segment
        segments = []
        segments.append(
            list(itertools.islice(self.buffer, start, relevant_done_indices[0]))
        )

        for i in range(len(relevant_done_indices) - 1):
            segment = list(
                itertools.islice(
                    self.buffer, relevant_done_indices[i], relevant_done_indices[i + 1]
                )
            )
            segments.append(segment)

        segments.append(
            list(itertools.islice(self.buffer, relevant_done_indices[-1], end))
        )

        return max(segments, key=len)


class ReservoirBuffer:
    """Maintains a representative sample of fixed size using reservoir sampling."""

    def __init__(self, capacity: int):
        self.capacity = capacity
        self.buffer = []
        self.count = 0

    def add(self, item):
        """Add item using reservoir sampling algorithm."""
        if len(self.buffer) < self.capacity:
            self.buffer.append(item)
        else:
            # Randomly replace elements with decreasing probability
            j = random.randint(0, self.count)
            if j < self.capacity:
                self.buffer[j] = item
        self.count += 1

    def get_samples(self):
        """Return current buffer contents."""
        return self.buffer


class BalancedStreamingKMeans:
    """Memory-efficient K-means clustering using reservoir sampling for balanced representation."""

    def __init__(self, n_clusters: int, reservoir_size: int, batch_size: int):
        self.n_clusters = n_clusters
        self.batch_size = batch_size
        self.kmeans = None
        self.reservoir = ReservoirBuffer(reservoir_size)
        self.reward_buffer = ReservoirBuffer(reservoir_size)

    def partial_fit(self, X: np.ndarray, rewards: np.ndarray):
        """Update reservoir with new batch of data."""
        for x, r in zip(X, rewards):
            self.reservoir.add(x)
            self.reward_buffer.add(r)

    def finalize(self):
        """Perform final clustering on reservoir samples."""
        X = np.array(self.reservoir.get_samples())
        rewards = np.array(self.reward_buffer.get_samples())

        if len(X) > 0:
            self.kmeans = MiniBatchKMeans(
                n_clusters=min(self.n_clusters, len(X)),
                batch_size=min(self.batch_size, len(X)),
                random_state=42,
            )
            labels = self.kmeans.fit_predict(X)

            # Compute cluster statistics
            cluster_sums = np.zeros((self.n_clusters, X.shape[1]))
            cluster_counts = np.zeros(self.n_clusters)
            reward_sums = np.zeros(self.n_clusters)
            reward_counts = np.zeros(self.n_clusters)

            for i in range(self.n_clusters):
                mask = labels == i
                cluster_sums[i] = X[mask].sum(axis=0)
                cluster_counts[i] = mask.sum()
                reward_sums[i] = rewards[mask].sum()
                reward_counts[i] = mask.sum()

            # Compute final representatives and rewards
            valid_clusters = cluster_counts > 0
            representatives = (
                cluster_sums[valid_clusters]
                / cluster_counts[valid_clusters, np.newaxis]
            )
            avg_rewards = reward_sums[valid_clusters] / reward_counts[valid_clusters]

            return representatives, avg_rewards
        return np.array([]), np.array([])


def generate_feedback_stream(
    model_class: Type[Union[PPO, SAC]],
    expert_models: List[Union[PPO, SAC]],
    environment: gym.Env,
    environment_name: str,
    total_steps: int,
    n_feedback: int,
    segment_len: int,
    checkpoints_dir: str,
    algorithm: str,
    batch_size: int = 1000,
) -> dict:
    # Initialize buffers for streaming
    segment_buffer = SegmentBuffer(maxlen=segment_len * 2)
    best_segments_reservoir = ReservoirBuffer(capacity=n_feedback * 2)
    best_demos_reservoir = ReservoirBuffer(capacity=n_feedback * 2)

    streaming_kmeans = BalancedStreamingKMeans(
        n_clusters=n_feedback,
        reservoir_size=min(10000, n_feedback * 20),
        batch_size=batch_size,
    )

    gamma = expert_models[0][0].gamma
    checkpoint_files = ["random"] + sorted(
        [f for f in os.listdir(checkpoints_dir) if re.search(r"rl_model_.*\.zip", f)],
        key=lambda x: int(re.search(r"\d+", x).group()) if re.search(r"\d+", x) else 0,
    )

    for model_file in checkpoint_files:
        if model_file != "random":
            model = model_class.load(
                os.path.join(checkpoints_dir, model_file),
                custom_objects={"learning_rate": 0.0, "lr_schedule": lambda _: 0.0},
            )
            norm_env_path = os.path.join(
                checkpoints_dir, environment_name, "vecnormalize.pkl"
            )
            norm_env = (
                VecNormalize.load(norm_env_path, DummyVecEnv([lambda: environment]))
                if os.path.isfile(norm_env_path)
                else None
            )
        else:
            model = None
            norm_env = None

        observation, _ = environment.reset()
        steps_this_checkpoint = total_steps // len(checkpoint_files)

        for step in range(steps_this_checkpoint):
            if model is not None:
                actions = model.predict(
                    norm_env.normalize_obs(observation) if norm_env else observation,
                    deterministic=True,
                )[0]
            else:
                actions = environment.action_space.sample()

            next_observation, reward, terminated, truncated, _ = environment.step(
                actions
            )
            done = terminated or truncated

            segment_buffer.append(
                (np.expand_dims(observation, axis=0), actions, reward, done), step, done
            )

            if step % segment_len == 0 and step > 0:
                # Get segment from buffer
                segment = segment_buffer.get_segment(step - segment_len, segment_len)
                if len(segment) >= segment_len // 2:  # Minimum length requirement
                    # Calculate return for the segment
                    segment_return = sum(
                        s[2] * (gamma**i) for i, s in enumerate(segment)
                    )
                    best_segments_reservoir.add((segment, segment_return))

                    # Try to generate better demonstration
                    state = environment.save_state(observation=segment[0][0].squeeze(0))
                    best_demo = None
                    best_demo_return = float("-inf")

                    # Try each expert model
                    for expert_model, exp_norm_env in expert_models:
                        _, _ = environment.reset()
                        obs = environment.load_state(state)

                        current_demo = []
                        current_return = 0

                        for t in range(segment_len):
                            action = expert_model.predict(
                                (
                                    exp_norm_env.normalize_obs(obs)
                                    if exp_norm_env
                                    else obs
                                ),
                                deterministic=True,
                            )[0]
                            next_obs, rew, terminated, truncated, _ = environment.step(
                                action
                            )
                            current_return += rew * (gamma**t)
                            current_demo.append(
                                (
                                    np.expand_dims(obs, axis=0),
                                    action,
                                    rew,
                                    terminated or truncated,
                                )
                            )

                            if terminated or truncated:
                                break
                            obs = next_obs

                        if current_return > best_demo_return:
                            best_demo_return = current_return
                            best_demo = current_demo

                    # Only keep demonstration if it improves upon original segment
                    if best_demo_return > segment_return:
                        best_demos_reservoir.add(
                            (best_demo, segment, best_demo_return - segment_return)
                        )

                    # Add to clustering
                    obs = np.array(
                        [
                            np.concatenate(
                                (
                                    s[0].squeeze(0).flatten(),
                                    np.expand_dims(s[1], 0) if s[1].ndim == 0 else s[1],
                                )
                            )
                            for s in segment
                        ]
                    )
                    rewards = np.array([s[2] for s in segment])
                    streaming_kmeans.partial_fit(obs, rewards)

            observation = next_observation if not done else environment.reset()[0]

    # Get final segments and demos from reservoirs
    reservoir_segments = best_segments_reservoir.get_samples()
    reservoir_demos = best_demos_reservoir.get_samples()

    # Sort all by improvement
    improvements = []
    for i, (seg, ret) in enumerate(reservoir_segments):
        # Find corresponding demo if it exists
        demo_improvement = None
        for demo, orig_seg, improvement in reservoir_demos:
            if str(orig_seg) == str(seg):
                demo_improvement = improvement
                break
        improvements.append(
            (i, demo_improvement if demo_improvement is not None else 0)
        )

    # Sort indices by improvement
    sorted_indices = [
        i for i, _ in sorted(improvements, key=lambda x: x[1], reverse=True)
    ]

    final_demos = []
    final_corrections = []
    segments = []
    opt_gaps = []
    selected_indices = []

    # First pass: take segments with corrections
    for idx in sorted_indices:
        if len(final_demos) >= n_feedback:
            break
        seg, ret = reservoir_segments[idx]
        demo_found = False
        for demo, orig_seg, _ in reservoir_demos:
            if str(orig_seg) == str(seg):
                final_demos.append(demo)
                final_corrections.append((seg, demo))
                segments.append(seg)
                opt_gaps.append(-ret)
                selected_indices.append(idx)
                demo_found = True
                break

    # Generate ratings using equal-width binning
    max_rating = 10
    ratings = max_rating - np.digitize(
        opt_gaps, np.linspace(min(opt_gaps), max(opt_gaps), max_rating)
    )

    # Generate preferences
    preferences = []
    tolerance = np.std(opt_gaps) / 10.0
    for _ in range(n_feedback):
        i, j = random.sample(range(len(segments)), 2)
        if abs(opt_gaps[i] - opt_gaps[j]) > tolerance:
            preferences.append((i, j, 1) if opt_gaps[i] > opt_gaps[j] else (j, i, 1))

    # Finalize clustering and generate descriptions
    cluster_representatives, cluster_rewards = streaming_kmeans.finalize()

    obs_dim = segments[0][0][0].squeeze(0).shape[0]
    cluster_descriptions = [
        (rep[:obs_dim], rep[obs_dim:], reward)
        for rep, reward in zip(cluster_representatives, cluster_rewards)
    ]

    # Generate description preferences
    descr_preferences = []
    tolerance = np.std(cluster_rewards) / 10.0
    for _ in range(n_feedback):
        i, j = random.sample(range(len(cluster_descriptions)), 2)
        if abs(cluster_rewards[i] - cluster_rewards[j]) > tolerance:
            descr_preferences.append(
                (i, j, 1) if cluster_rewards[i] < cluster_rewards[j] else (j, i, 1)
            )

    return {
        "segments": segments,
        "ratings": ratings.tolist(),
        "preferences": preferences,
        "demos": demos,
        "corrections": corrections,
        "description": cluster_descriptions,
        "description_preference": descr_preferences,
        "opt_gaps": opt_gaps,
    }


def setup_environment(environment_name: str) -> gym.Env:
    """Setup environment based on environment type."""
    if "procgen" in environment_name:
        _, short_name, _ = environment_name.split("-")
        environment = Gym3ToGymnasium(ProcgenGym3Env(num=1, env_name=short_name))
        environment = SaveResetEnvWrapper(
            TransformObservation(
                environment, lambda obs: obs["rgb"], environment.observation_space
            )
        )

    elif "ALE/" in environment_name:
        environment = FrameStackObservation(WarpFrame(gym.make(environment_name)), 4)
        environment = SaveResetEnvWrapper(
            TransformObservation(
                environment, lambda obs: obs.squeeze(-1), environment.observation_space
            )
        )

    elif "MiniGrid" in environment_name:
        environment = SaveResetEnvWrapper(FlatObsWrapper(gym.make(environment_name)))

    else:
        environment = SaveResetEnvWrapper(gym.make(environment_name))

    return environment


def load_expert_models(
    args, environment: gym.Env, checkpoints_path: str
) -> List[Tuple[Union[PPO, SAC], VecNormalize]]:
    """Load expert models based on evaluation scores."""
    env_name = (
        args.environment
        if "ALE" not in args.environment
        else args.environment.replace("/", "-")
    )
    expert_model_paths = [
        os.path.join(checkpoints_path, args.algorithm, model)
        for model in os.listdir(os.path.join(checkpoints_path, args.algorithm))
        if env_name in model
    ]

    try:
        # Load evaluation scores and select top N models
        run_eval_scores = pd.read_csv(
            os.path.join(checkpoints_path, "collected_results.csv")
        )
        run_eval_scores = (
            run_eval_scores.loc[run_eval_scores["env"] == args.environment]
            .sort_values(by=["eval_score"], ascending=False)
            .head(args.top_n_models)["run"]
            .to_list()
        )
        expert_model_paths = [
            path
            for path in expert_model_paths
            if path.split(os.path.sep)[-1] in run_eval_scores
        ]
    except:
        print(
            "[WARN] No eval benchmark results are available. Check your eval benchmarks"
        )

    expert_models = []
    for expert_model_path in expert_model_paths:
        # Load normalization environment if available
        if os.path.isfile(
            os.path.join(expert_model_path, env_name, "vecnormalize.pkl")
        ):
            norm_env = VecNormalize.load(
                os.path.join(expert_model_path, env_name, "vecnormalize.pkl"),
                DummyVecEnv([lambda: environment]),
            )
        else:
            norm_env = None

        # Load appropriate model based on environment type
        if "ALE" not in env_name:
            model = (PPO if args.algorithm == "ppo" else SAC).load(
                os.path.join(expert_model_path, f"{env_name}.zip")
            )
        else:
            model = (PPO if args.algorithm == "ppo" else SAC).load(
                os.path.join(expert_model_path, f"best_model.zip")
            )

        expert_models.append((model, norm_env))

    return expert_models


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", type=int, default=0)
    parser.add_argument("--algorithm", type=str, default="ppo")
    parser.add_argument("--environment", type=str, default="HalfCheetah-v5")
    parser.add_argument("--n-steps-factor", type=int, default=20)
    parser.add_argument("--n-feedback", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--segment-len", type=int, default=50)
    parser.add_argument("--save-folder", type=str, default="feedback")
    parser.add_argument("--top-n-models", type=int, default=3)
    args = parser.parse_args()

    np.random.seed(args.seed)
    random.seed(args.seed)

    env_name = (
        args.environment
        if "ALE" not in args.environment
        else args.environment.replace("/", "-")
    )
    feedback_id = f"{args.algorithm}_{env_name}"
    feedback_path = (
        Path(__file__).parents[1].resolve()
        / args.save_folder
        / f"{feedback_id}_{args.seed}.pkl"
    )
    checkpoints_path = "../main/gt_agents"

    environment = setup_environment(args.environment)
    expert_models = load_expert_models(args, environment, checkpoints_path)

    model_class = PPO if args.algorithm == "ppo" else SAC

    feedback = generate_feedback_stream(
        model_class=model_class,
        expert_models=expert_models,
        environment=environment,
        environment_name=args.environment,
        total_steps=args.n_feedback * args.n_steps_factor,
        n_feedback=args.n_feedback,
        segment_len=args.segment_len,
        checkpoints_dir=os.path.join(checkpoints_path, args.algorithm, f"{env_name}_1"),
        algorithm=args.algorithm,
    )

    feedback_path.parent.mkdir(parents=True, exist_ok=True)
    with open(feedback_path, "wb") as feedback_file:
        pickle.dump(feedback, feedback_file, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":
    main()
