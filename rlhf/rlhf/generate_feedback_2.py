import argparse
import os
import pickle
import re
from pathlib import Path
from typing import Type, Union, List

import gymnasium as gym
from gymnasium.wrappers.stateful_observation import FrameStackObservation
from gymnasium.wrappers.transform_observation import TransformObservation
from minigrid.wrappers import FlatObsWrapper 
from procgen import ProcgenGym3Env
from rl_zoo3.wrappers import Gym3ToGymnasium
# necessary to import ale_py/procgen, otherwise it will not be found
import ale_py
import procgen
import numpy as np
from sklearn.cluster import MiniBatchKMeans
import torch
import itertools
from numpy.typing import NDArray
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.atari_wrappers import WarpFrame
from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv
from torch import Tensor
import random
import itertools
import bisect

import pandas as pd

from rlhf.datatypes import FeedbackDataset
from rlhf.save_reset_wrapper import SaveResetEnvWrapper


def predict_expert_value(
    expert_model: Union[PPO, SAC], 
    observation: np.ndarray, 
    actions: Tensor = None
) -> Tensor:
    """Return the value from the expert's value function for a given observation and actions."""

    expert_model, norm_env = expert_model

    if norm_env is not None:
        observation = norm_env.normalize_obs(observation)
    
    observation = expert_model.policy.obs_to_tensor(observation)[0]
    with torch.no_grad():
        return torch.mean(
            torch.cat(expert_model.policy.critic_target(observation, actions), dim=1) if isinstance(expert_model, SAC) else expert_model.policy.predict_values(observation),
            dim=1,
            keepdim=True,
        )[0]

def create_segments(arr, start_indices, done_indices, segment_length, min_segment_len):
    """
        Creates array segments with target length (segment_length) and minimum length min_segment_len,
        selects the longest contiguous array within a segment [start_indices: start_indeces+segment_length]
    """
    segments = []
    
    for start in start_indices:
        end = start + segment_length
        
        # Find the position of the first done_index greater than or equal to start
        insert_pos = bisect.bisect_left(done_indices, start)
        
        # Collect all done_indices within the range [start, end)
        relevant_done_indices = []
        while insert_pos < len(done_indices) and done_indices[insert_pos] < end:
            relevant_done_indices.append(done_indices[insert_pos])
            insert_pos += 1
        
        # If there are no relevant done_indices, take the full segment
        if not relevant_done_indices:
            segment = arr[start:end]
        else:
            # Consider the segment before the first done_index
            longest_segment = arr[start:relevant_done_indices[0]]
            
            # Consider segments between each pair of consecutive done_indices
            for i in range(len(relevant_done_indices) - 1):
                segment = arr[relevant_done_indices[i]:relevant_done_indices[i + 1]]
                if len(segment) > len(longest_segment):
                    longest_segment = segment
            
            # Consider the segment after the last done_index
            segment = arr[relevant_done_indices[-1]:end]
            if len(segment) > len(longest_segment):
                longest_segment = segment
            
            # Use the longest valid segment
            segment = longest_segment
        
        if len(segment) > min_segment_len:  # Only add non-empty segments
            segments.append(segment)
    
    return segments

def discounted_sum_numpy(rewards, gamma):
    return np.sum(rewards * (gamma ** np.arange(len(rewards))))

def equal_depth_binning_with_indices(data, num_bins):
    data = np.array(data)
    # Sort the data and get the original indices
    sorted_indices = np.argsort(data)
    sorted_data = np.sort(data)
    
    # Determine the number of elements per bin
    bin_size = len(data) // num_bins
    remainder = len(data) % num_bins
    
    bins = []
    bin_indices = np.zeros(len(data), dtype=int)
    start = 0
    
    for i in range(num_bins):
        end = start + bin_size + (1 if i < remainder else 0)
        bin_indices[sorted_indices[start:end]] = i
        bins.append(sorted_data[start:end])
        start = end
    
    return bin_indices, bins

def equal_width_binning_with_indices(data, num_bins):
    data = np.array(data)
    # Find the minimum and maximum values in the data
    min_val, max_val = np.min(data), np.max(data)

    # Create bin edges
    bin_edges = np.linspace(min_val, max_val, num_bins + 1)
    
    # Use numpy's digitize function to assign bin indices
    bin_indices = np.digitize(data, bin_edges[:-1])
    
    # Create the bins
    bins = [data[(bin_indices == i)] for i in range(1, num_bins + 1)]
    
    return bin_indices, bins

def get_preference_pairs(segments, opt_gaps, n_feedback, tolerance=0.25):
    all_pairs = list(enumerate(itertools.combinations(range(len(segments)), 2)))
    random.shuffle(all_pairs)
    
    preferences = []
    sampled_indices = []
    
    for idx, (a, b) in all_pairs:
        gap_diff = opt_gaps[a] - opt_gaps[b]
        if abs(gap_diff) > tolerance:
            if gap_diff > 0:
                preferences.append((a, b, 1))
            else:
                preferences.append((b, a, 1))
            sampled_indices.append(idx)
            
            if len(preferences) == n_feedback:
                break
    
    if len(preferences) < n_feedback:
        raise ValueError(f"Could only generate {len(preferences)} preferences with the given tolerance. Increase the number of segments, decrease the tolerance, or decrease n_feedback.")
    
    return preferences

def get_preference_pairs_descript(clusters, rews, n_feedback, tolerance=0.25):
    all_pairs = list(enumerate(itertools.combinations(range(len(clusters)), 2)))
    random.shuffle(all_pairs)
    
    preferences = []
    sampled_indices = []
    
    for idx, (a, b) in all_pairs:
        rew_diff = rews[a] - rews[b]
        if abs(rew_diff) > tolerance:
            if rew_diff < 0:
                preferences.append((a, b, 1))
            else:
                preferences.append((b, a, 1))
            sampled_indices.append(idx)
            
            if len(preferences) == n_feedback:
                break
    
    if len(preferences) < n_feedback:
        raise ValueError(f"Could only generate {len(preferences)} preferences with the given tolerance. Increase the number of segments, decrease the tolerance, or decrease n_feedback.")
    
    return preferences

def debug_feedback_output(feedback_data):
    print("\nDebugging Feedback Output:")
    print(f"Number of segments: {len(feedback_data['segments'])}")
    print(f"Number of ratings: {len(feedback_data['ratings'])}")
    print(f"Number of preferences: {len(feedback_data['preferences'])}")
    print(f"Number of demos: {len(feedback_data['demos'])}")
    print(f"Number of corrections: {len(feedback_data['corrections'])}")
    print(f"Number of cluster descriptions: {len(feedback_data['description'])}")
    print(f"Number of description preferences: {len(feedback_data['description_preference'])}")
    print(f"Number of optimality gaps: {len(feedback_data['opt_gaps'])}")
    
    # Additional checks
    print("\nConsistency Checks:")
    n_feedback = len(feedback_data['segments'])
    print(f"All main feedback arrays have {n_feedback} elements: {all(len(feedback_data[key]) == n_feedback for key in ['segments', 'ratings', 'demos', 'corrections', 'opt_gaps'])}")
    
    # Check segment lengths
    segment_lengths = [len(segment) for segment in feedback_data['segments']]
    print(f"Minimum segment length: {min(segment_lengths)}")
    print(f"Maximum segment length: {max(segment_lengths)}")
    print(f"Average segment length: {sum(segment_lengths) / len(segment_lengths):.2f}")

def generate_feedback(
    model_class: Type[Union[PPO, SAC]],
    expert_models: List[Union[PPO, SAC]],
    environment: gym.Env,
    environment_name: str = "HalfCheetah-v5",
    checkpoints_path: str = "rl_checkpoints",
    total_steps_factor: int = 50,
    n_feedback: int = 100,
    segment_len: int = 50,
    min_segment_len: int = 25,
    algorithm: str = "sac",
    device: str = "cuda",
    binning_type: str = "width"
) -> dict:
    """Generate agent's observations and feedback in the training environment."""
    feedback_id = f"{algorithm}_{environment_name.replace('/', '-')}"
    checkpoints_dir = os.path.join(checkpoints_path, algorithm, f"{environment_name.replace('/', '-')}_1")

    print(f"Generating feedback for: {feedback_id}")

    # Adaptive oversampling
    oversampling_factor = 1.5
    target_n_feedback = int(n_feedback * oversampling_factor)

    checkpoint_files = [
        file for file in os.listdir(checkpoints_dir) if re.search(r"rl_model_.*\.zip", file)
    ] or [f"{environment_name}.zip"]    

    total_steps = n_feedback * total_steps_factor
    num_checkpoints = len(checkpoint_files) + 1
    steps_per_checkpoint = total_steps // num_checkpoints
    feedback_per_checkpoint = target_n_feedback // num_checkpoints
    gamma = expert_models[0][0].gamma

    checkpoint_files = ["random"] + sorted(checkpoint_files, key=lambda x: int(re.search(r'\d+', x).group()))

    print(f"""
    Feedback Generation Debug Info:
      Feedback ID: {feedback_id}
      Checkpoints Directory: {checkpoints_dir}
      Number of Checkpoints: {num_checkpoints}
      Checkpoint Files: {checkpoint_files}
      Total Steps: {total_steps}
      Steps per Checkpoint: {steps_per_checkpoint}
      Target Feedback: {n_feedback}
      Oversampled Generated Feedback Instances: {target_n_feedback}
      Feedback per Checkpoint: {feedback_per_checkpoint}
      Env. Gamma: {gamma}
    """)

    segments = []
    state_copies = []
    for model_file in checkpoint_files:
        feedback = []
        fb_indices = random.sample(range(steps_per_checkpoint - segment_len + 1), k=feedback_per_checkpoint)
        final_segment_indices = sorted(set(fb_indices))
  
        if model_file != "random":
            model = model_class.load(
                os.path.join(checkpoints_dir, model_file),
                custom_objects={"learning_rate": 0.0, "lr_schedule": lambda _: 0.0},
            )
            norm_env_path = os.path.join(checkpoints_dir, environment_name, "vecnormalize.pkl")
            norm_env = VecNormalize.load(norm_env_path, DummyVecEnv([lambda: environment])) if os.path.isfile(norm_env_path) else None
        else:
            model = None
            norm_env = None

        observation, _ = environment.reset()

        for step in range(steps_per_checkpoint):
            if step in final_segment_indices:
                state_copies.append(environment.save_state(observation=observation))

            if model is not None:
                actions, _ = model.predict(norm_env.normalize_obs(observation) if norm_env else observation, deterministic=True)
            else:
                actions = environment.action_space.sample()
            
            next_observation, reward, terminated, truncated, _ = environment.step(actions)
            done = terminated or truncated

            feedback.append((np.expand_dims(observation, axis=0), actions, reward, done))

            observation = next_observation if not done else environment.reset()[0]

        segments.extend(create_segments(feedback, final_segment_indices, np.where([f[3] for f in feedback])[0], segment_len, min_segment_len))

        print(f"Generated segments: {len(segments)} of target {target_n_feedback}")
    
    opt_gaps = []
    for seg in segments:
        initial_vals = [predict_expert_value(expert_model, np.array(seg[0][0])).item() for expert_model in expert_models]
        initial_val = np.mean(initial_vals)

        discounted_rew_sum = discounted_sum_numpy([s[2] for s in seg[:-1]], gamma)
        
        final_vals = [predict_expert_value(expert_model, np.array(seg[-1][0])).item() for expert_model in expert_models]
        final_val = np.mean(final_vals)

        opt_gap = (initial_val - gamma ** len(seg) * final_val) - discounted_rew_sum
        opt_gaps.append(opt_gap)
    
    max_rating = 10
    ratings = max_rating - (equal_width_binning_with_indices(opt_gaps, max_rating)[0] if binning_type == "width" else equal_depth_binning_with_indices(opt_gaps, max_rating)[0])

    print("[INFO] Successfully generated evaluative feedback")

    demos = []
    corrections = []
    improvements = []
    
    for i, state in enumerate(state_copies):
        current_demos = []
        current_expert_model_returns = []
        
        for exp_model_index, (expert_model, exp_norm_env) in enumerate(expert_models):
            _, _ = environment.reset()
            obs = environment.load_state(state)
    
            demo = []
            for _ in range(segment_len):
                action, _ = expert_model.predict(exp_norm_env.normalize_obs(obs) if exp_norm_env else obs, deterministic=True)
                new_obs, rew, terminated, truncated, _ = environment.step(action)
                done = terminated or truncated
                demo.append((np.expand_dims(obs, axis=0), action, rew, done, exp_model_index))
                obs = new_obs
    
                if done:
                    break
    
            current_demos.append(demo)
            current_expert_model_returns.append(discounted_sum_numpy([d[2] for d in demo], gamma))
    
        best_index = np.argmax(current_expert_model_returns)
        best_demo = current_demos[best_index]
        best_demo_return = current_expert_model_returns[best_index]
    
        original_return = discounted_sum_numpy([s[2] for s in segments[i]], gamma)
    
        if best_demo_return > original_return:
            demos.append(best_demo)
            corrections.append((segments[i], best_demo))
            improvements.append(best_demo_return - original_return)
        else:
            demos.append(None)
            corrections.append(None)
            improvements.append(0)
    
    sorted_indices = np.argsort(improvements)[::-1]
    
    final_demos = []
    final_corrections = []
    selected_indices = []
    for idx in sorted_indices:
        if len(final_demos) >= n_feedback:
            break
        if corrections[idx] is not None:
            final_demos.append(demos[idx])
            final_corrections.append(corrections[idx])
            selected_indices.append(idx)
    
    if len(final_demos) < n_feedback:
        for idx in sorted_indices:
            if len(final_demos) >= n_feedback:
                break
            if idx not in selected_indices:
                final_demos.append(demos[idx])
                final_corrections.append((segments[idx], demos[idx]))
                selected_indices.append(idx)

    # Use selected_indices to sample segments, ratings, and opt_gaps
    segments = [segments[i] for i in selected_indices]
    ratings = [ratings[i] for i in selected_indices]
    opt_gaps = [opt_gaps[i] for i in selected_indices]

    # now we can sample the pairs after we have pruned segments
    tolerance = np.std(opt_gaps) / 10.
    preferences = get_preference_pairs(segments, opt_gaps, n_feedback, tolerance=tolerance)

    print("[INFO] Successfully generated comparative feedback")

    demos = final_demos
    corrections = final_corrections

    print("[INFO] Successfully generated demonstrative/corrective feedback")

    # The rest of the clustering and description generation code remains the same
    all_obs = []
    all_rewards = []
    for seg in segments:
        obs = np.array([np.concatenate((s[0].squeeze(0), s[1])) for s in seg])
        rewards = np.array([s[2] for s in seg])
        all_obs.append(obs)
        all_rewards.append(rewards)
    states = np.concatenate(all_obs, axis=0)
    rewards = np.concatenate(all_rewards, axis=0)

    batch_size = min(1000, len(states) // 100)  # Adaptive batch size
    kmeans = MiniBatchKMeans(n_clusters=n_feedback, batch_size=batch_size, random_state=42)
    kmeans.fit(states)
    cluster_assignments = kmeans.predict(states)

    cluster_representatives = []
    cluster_rewards = []
    for i in range(n_feedback):
        cluster_mask = cluster_assignments == i
        cluster_states = states[cluster_mask]
        cluster_state_rewards = rewards[cluster_mask]
        if len(cluster_states) > 0 and not np.any(np.isnan(np.mean(cluster_states, axis=0))):
            cluster_representatives.append(np.mean(cluster_states, axis=0))
            cluster_rewards.append(np.mean(cluster_state_rewards))
    cluster_representatives = np.array(cluster_representatives)
    cluster_rewards = np.array(cluster_rewards)

    obs_dim = segments[0][0][0].squeeze(0).shape[0]
    cluster_descriptions = [
        (rep[:obs_dim], rep[obs_dim:], reward) 
        for rep, reward in zip(cluster_representatives, cluster_rewards)
    ]
    tolerance = np.std(cluster_rewards) / 10.
    descr_preferences = get_preference_pairs_descript(cluster_descriptions, cluster_rewards, n_feedback, tolerance=tolerance)

    print("[INFO] Successfully generated descriptive feedback")
    
    return {
        "segments": segments,
        "ratings": ratings,
        "preferences": preferences,
        "demos": demos,
        "corrections": corrections,
        "description": cluster_descriptions,
        "description_preference": descr_preferences,
        "opt_gaps": opt_gaps
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", type=int, default=0, help="Experiment number")
    parser.add_argument("--algorithm", type=str, default="ppo", help="RL algorithm")
    parser.add_argument("--environment", type=str, default="HalfCheetah-v5", help="Environment")
    parser.add_argument("--n-steps-factor", type=int, default=int(20), help="Number of steps sampled for each feedback instance")
    parser.add_argument("--n-feedback", type=int, default=int(1000), help="How many feedback instances should be generated")
    parser.add_argument("--seed", type=int, default=1337, help="TODO: Seed for env and stuff")
    parser.add_argument("--segment-len", type=int, default=50, help="How long is the segment we generate feedback for")
    parser.add_argument("--save-folder", type=str, default="feedback", help="Where to save the feedback")
    parser.add_argument("--top-n-models", type=int, default=4)
    args = parser.parse_args()

    np.random.seed(args.seed)
    random.seed(args.seed)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    feedback_id = f"{args.algorithm}_{args.environment}"
    feedback_path = Path(__file__).parents[1].resolve() / args.save_folder / f"{feedback_id}_{args.seed}.pkl"
    checkpoints_path = "../main/gt_agents"

    # load "ensemble" of expert agents
    env_name = args.environment if "ALE" not in args.environment else args.environment.replace("/","-")
    expert_model_paths = [os.path.join(checkpoints_path, args.algorithm, model) for model in os.listdir(os.path.join(checkpoints_path, args.algorithm)) if env_name in model]
    print(expert_model_paths)
    orig_len = len(expert_model_paths)
    #expert_model = (PPO if args.algorithm == "ppo" else SAC).load(
    #    os.path.join(checkpoints_path, args.algorithm, f"{args.environment.replace("/", "-")}_1", "best_model.zip")
    #)


    try:
        run_eval_scores = pd.read_csv(os.path.join(checkpoints_path, "collected_results.csv"))
        run_eval_scores = run_eval_scores.loc[run_eval_scores['env'] == args.environment].sort_values(by=['eval_score'], ascending=False).head(args.top_n_models)["run"].to_list()
        expert_model_paths = [path for path in expert_model_paths if path.split(os.path.sep)[-1] in run_eval_scores]
    except:
        print("[WARN] No eval benchmark results are available. Check you eval benchmarks")    

    if "procgen" in args.environment:
        _, short_name, _ = args.environment.split("-")
        environment = Gym3ToGymnasium(ProcgenGym3Env(num=1, env_name=short_name))
        environment = SaveResetEnvWrapper(TransformObservation(environment, lambda obs: obs["rgb"], environment.observation_space))
    elif "ALE/" in args.environment:
        environment = FrameStackObservation(WarpFrame(gym.make(args.environment)), 4)
        environment = SaveResetEnvWrapper(TransformObservation(environment, lambda obs: obs.squeeze(-1), environment.observation_space))
    elif "MiniGrid" in args.environment:
        environment = SaveResetEnvWrapper(FlatObsWrapper(gym.make(args.environment)))
    else:
        environment = SaveResetEnvWrapper(gym.make(args.environment))

    expert_models = []
    for expert_model_path in expert_model_paths:
        if os.path.isfile(os.path.join(expert_model_path, env_name, "vecnormalize.pkl")):
            norm_env = VecNormalize.load(os.path.join(expert_model_path, env_name, "vecnormalize.pkl"), DummyVecEnv([lambda: environment]))
        else:
            norm_env = None
        expert_models.append(((PPO if args.algorithm == "ppo" else SAC).load(os.path.join(expert_model_path, "best_model.zip")
        ), norm_env))
    
    model_class = PPO if args.algorithm == "ppo" else SAC

    feedback = generate_feedback(
        model_class,
        expert_models,
        environment,
        environment_name=args.environment,
        total_steps_factor=args.n_steps_factor,
        n_feedback=args.n_feedback,
        segment_len=args.segment_len,
        checkpoints_path=checkpoints_path,
        algorithm=args.algorithm,
        device=device,
    )

    debug_feedback_output(feedback)

    feedback_path.parent.mkdir(parents=True, exist_ok=True)
    with open(feedback_path, "wb") as feedback_file:
        print("FB path", feedback_path)
        pickle.dump(feedback, feedback_file, protocol=pickle.HIGHEST_PROTOCOL)

if __name__ == "__main__":
    main()
