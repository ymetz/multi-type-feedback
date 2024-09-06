import argparse
import os
import pickle
import re
from pathlib import Path
from typing import Type, Union, List
import matplotlib.pyplot as plt

import gymnasium as gym
from gymnasium.wrappers.stateful_observation import FrameStackObservation
from gymnasium.wrappers.transform_observation import TransformObservation
from minigrid.wrappers import FlatObsWrapper 
from procgen import ProcgenGym3Env
from rl_zoo3.wrappers import Gym3ToGymnasium
import ale_py
import minigrid
# minigrid.register_minigrid_envs()
import numpy as np
import torch
from captum.attr import IntegratedGradients
from numpy.typing import NDArray
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.vec_env import VecExtractDictObs
from stable_baselines3.common.atari_wrappers import AtariWrapper
from torch import Tensor
import random
import itertools
import bisect

from rlhf.datatypes import FeedbackDataset
from rlhf.save_reset_wrapper import SaveResetEnvWrapper

def get_attributions(
    observation: Tensor,
    actions: NDArray,
    explainer: IntegratedGradients,
    algorithm: str = "sac",
    device: str = "cuda",
) -> NDArray[np.float64]:
    """
    Compute attributions for a given observation and actions using the provided explainer.
    """
    obs_baselines = torch.zeros_like(observation)
    if algorithm == "sac":
        actions_tensor = torch.from_numpy(actions).unsqueeze(0).to(device)
        actions_baselines = torch.zeros_like(actions_tensor)
    
        attributions = explainer.attribute(
            (observation, actions_tensor),
            target=0,
            baselines=(obs_baselines, actions_baselines),
            internal_batch_size=64,
        )
    else:
        attributions = explainer.attribute(
            observation,
            target=0,
            baselines=obs_baselines,
            internal_batch_size=64,
        )        

    return attributions.squeeze().cpu().numpy()

def predict_expert_value(
    expert_model: Union[PPO, SAC], 
    observation: np.ndarray, 
    actions: Tensor = None
) -> Tensor:
    """Return the value from the expert's value function for a given observation and actions."""

    observation = expert_model.policy.obs_to_tensor(observation)[0]
    with torch.no_grad():
        return torch.min(
            torch.cat(expert_model.policy.critic_target(observation, actions), dim=1) if isinstance(expert_model, SAC) else expert_model.policy.predict_values(observation),
            dim=1,
            keepdim=True,
        )[0]

def get_model_logits(
    expert_model: Union[PPO, SAC],
    observation: Tensor,
    actions: Tensor = None,
) -> Tensor:
    if isinstance(expert_model, SAC):
        return torch.min(
            torch.cat(expert_model.policy.critic_target(observation, actions), dim=1) if isinstance(expert_model, SAC) else expert_model.policy.predict_values(observation),
            dim=1,
            keepdim=True,
        )[0]
    else:
        return expert_model.policy.predict_values(observation)

"""
archive
def create_segments(arr, start_indices, done_indices, segment_length):
    segments = []
    for start in start_indices:
        segment = arr[start:start + segment_length]
        segments.append(segment)
    return segments
"""

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

def discounted_sum_numpy(rewards, discount_factor):
    rewards = np.array(rewards)
    n = len(rewards)
    discount_factors = discount_factor ** np.arange(n)
    return np.sum(rewards * discount_factors)

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
    
    # Calculate the width of each bin
    bin_width = (max_val - min_val) / num_bins
    
    # Create bin edges
    bin_edges = np.linspace(min_val, max_val, num_bins + 1)
    
    # Use numpy's digitize function to assign bin indices
    bin_indices = np.digitize(data, bin_edges[:-1])
    
    # Create the bins
    bins = [data[(bin_indices == i)] for i in range(1, num_bins + 1)]
    
    return bin_indices, bins

def get_k_random_pairs(data, k):
    all_pairs = list(itertools.combinations(data, 2))
    if k > len(all_pairs):
        raise ValueError("k is too large for the number of possible unique pairs")
    return random.sample(all_pairs, k)

def generate_feedback(
    model_class: Type[Union[PPO, SAC]],
    expert_models: List[Union[PPO, SAC]],
    environment: gym.Env,
    environment_name: str = "HalfCheetah-v3",
    checkpoints_path: str = "rl_checkpoints",
    total_steps_factor: int = 50,
    n_feedback: int = 100,
    segment_len: int = 50,
    min_segment_len: int = 25, # segments can be pruned at the beginning or end of episodes, remove if shorter than min_len
    algorithm: str = "sac",
    device: str = "cuda",
    binning_type: str = "width"
) -> FeedbackDataset:
    """Generate agent's observations and feedback in the training environment."""
    feedback_id = f"{algorithm}_{environment_name.replace('/', '-')}"
    checkpoints_dir = os.path.join(checkpoints_path, algorithm, f"{environment_name.replace('/', '-')}_1")

    print(f"Generating feedback for: {feedback_id}")

    checkpoint_files = [
        file for file in os.listdir(checkpoints_dir) if re.search(r"rl_model_.*\.zip", file)
    ] or [f"{environment_name}.zip"]

    total_steps = n_feedback * total_steps_factor # how many steps we want to generate to sample from, a natural choice is the segment length
    num_checkpoints = len(checkpoint_files) + 1 # also sample steps from random actios as the 0th checkpoint
    steps_per_checkpoint = total_steps // num_checkpoints
    feedback_per_checkpoint = n_feedback // num_checkpoints
    gamma = expert_models[0].gamma

    # Sort the files based on the extracted numerical value, makes handling and debugging a bit easier, is shuffled later anyways if necessary
    checkpoint_files = ["random"] + sorted(checkpoint_files, key=lambda x: int(re.search(r'\d+', x).group()))

    print(f"""
    Feedback Generation Debug Info:
      Feedback ID: {feedback_id}
      Checkpoints Directory: {checkpoints_dir}
      Number of Checkpoints: {num_checkpoints}
      Checkpoint Files: {checkpoint_files}
      Total Steps: {total_steps}
      Steps per Checkpoint: {steps_per_checkpoint}
      Total Feedback Instances: {n_feedback}
      Feedback per Checkpoint: {feedback_per_checkpoint}
      Env. Gamma: {gamma}
    """)

    explainer_cls = IntegratedGradients

    segments = []
    for model_file in checkpoint_files:
        
        feedback = []
        # we already sample the indices for the number of generated feedback instances/segments (last segment_len steps should 
        # not be sampled from)
        fb_indices = random.choices(range(0, steps_per_checkpoint + 1 - segment_len), k=feedback_per_checkpoint)
  
        if model_file != "random":
            model = model_class.load(
                os.path.join(checkpoints_dir, model_file),
                custom_objects={"learning_rate": 0.0, "lr_schedule": lambda _: 0.0},
            )
        else:
            model = None

        observation, _ = environment.reset()
        state_copies = []

        # now collect original data
        
        for step in range(steps_per_checkpoint):
            if step in fb_indices:
                state_copies.append(environment.save_state(observation=observation))

            if model is not None:
                actions, _ = model.predict(observation, deterministic=True)
            else:
                actions = environment.action_space.sample()
            next_observation, reward, terminated, truncated, _ = environment.step(actions)
            done = terminated | truncated

            feedback.append(
                (np.expand_dims(observation, axis=0), actions, reward, done)
            )

            observation = next_observation if not done else environment.reset()[0]


        # generate feedback from collected examples, split at given indices and dones
        final_segment_indices = list(set(fb_indices)) #| set(np.where([f[3] for f in feedback] == True)[0])
        segments.extend(create_segments(feedback, final_segment_indices, np.where(np.array([f[3] for f in feedback]) == True)[0], segment_len, min_segment_len))

        print(f"Generated segments: {len(segments)} of approx. {n_feedback}")

    
    # start by computing the evaluative fb. (for the comparative one, we just used samples segment pairs)
    opt_gaps = []
    for seg in segments:
        # predict the initial value
        initial_val = predict_expert_value(
                expert_model, np.array(seg[0][0])
            ).item()
        print("INITIAL VAL", initial_val)

        # sum the discounted rewards, don't add reward for last step because we use it to calculate final value
        discounted_rew_sum = discounted_sum_numpy([s[2] for s in seg[:-1]], gamma)
        print("DISC.REW.SUM", discounted_rew_sum)
        
        # get the final value
        final_val = predict_expert_value(expert_model, np.array(seg[-1][0])).item()
        print("FINAL VAL", final_val)

        # opt gap is the expected returns - actual returns
        opt_gap = (initial_val - gamma ** len(seg) * final_val) - discounted_rew_sum
        opt_gaps.append(opt_gap)
        print("DISC. FINAL VALUE", gamma ** len(seg) * final_val, gamma ** len(seg))
        print("OPT GAP", opt_gap)
    
    # bin indices, which we interpret as rating feedback, of course flipped because a low opt. gap is good
    max_rating = 10
    if binning_type == "width":
        ratings = max_rating - equal_width_binning_with_indices(opt_gaps, max_rating)[0]
    else:
        ratings = max_rating - equal_depth_binning_with_indices(opt_gaps, max_rating)[0]

    # generate pair preferences, with tolerance 1.0
    tolerance = 1.0
    pairs = get_k_random_pairs(np.arange(len(segments)), n_feedback)
    preferences = [(a,b,1) if (opt_gaps[a] - opt_gaps[b] > tolerance) else (b,a,1) if (opt_gaps[b] - opt_gaps[a] > tolerance) else (a,b,0) for a, b in pairs]
    
    # instructive feedback, reset env and run expert model for demos for each segment
    demos = []
    corrections = []
    for i, state in enumerate(state_copies):
        _, _ = environment.reset()
        obs = environment.load_state(state)

        demo = []
        for _ in range(segment_len):
            action, _ = expert_model.predict(obs, deterministic=True)
            obs, rew, terminated, _, _ = environment.step(action)
            demo.append((np.expand_dims(obs, axis=0), action, rew, terminated))

            if terminated:
                break
        demos.append(demo)
        corrections.append((segments[i], demo))

    # descriptive feedback, for now, compute attributions and the average over features
    if algorithm == "sac":
        explainer = explainer_cls(lambda obs, acts: get_model_logits(expert_model, obs, acts))
    else:
        explainer = explainer_cls(lambda obs: get_model_logits(expert_model, obs))

    descriptions = []
    for i, seg in enumerate(segments):
        attributions = get_attributions(observation = expert_model.policy.obs_to_tensor(np.array([s[0].squeeze(0) for s in seg]))[0], actions=None, explainer=explainer, algorithm=algorithm)
        saliency = np.std(attributions) / np.mean(attributions) * np.abs(np.mean(attributions))
        descriptions.append((saliency, opt_gaps[i]))

    descr_preferences = [(a,b,1) if (descriptions[a][1] - descriptions[b][1] > tolerance) else (b, a, 1) if (descriptions[b][1] - descriptions[a][1] > tolerance) else (a, b, 0) for a, b in pairs]        

    return {
        "segments": segments,
        "ratings": ratings,
        "preferences": preferences,
        "demos": demos,
        "corrections": corrections,
        "description": descriptions,
        "description_preference": descr_preferences,
        "opt_gaps": opt_gaps
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", type=int, default=0, help="Experiment number")
    parser.add_argument("--algorithm", type=str, default="sac", help="RL algorithm")
    parser.add_argument("--environment", type=str, default="HalfCheetah-v5", help="Environment")
    parser.add_argument("--n-steps-factor", type=int, default=int(20), help="Number of steps sampled for each feedback instance")
    parser.add_argument("--n-feedback", type=int, default=int(1000), help="How many feedback instances should be generated")
    parser.add_argument("--seed", type=int, default=1337, help="TODO: Seed for env and stuff")
    parser.add_argument("--segment-len", type=int, default=50, help="How long is the segment we generate feedback for")
    parser.add_argument("--save-folder", type=str, default="feedback", help="Where to save the feedback")
    args = parser.parse_args()

    np.random.seed(args.seed)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    feedback_id = f"{args.algorithm}_{args.environment}"
    feedback_path = Path(__file__).parents[1].resolve() / args.save_folder / f"{feedback_id}.pkl"
    checkpoints_path = "../main/logs"

    # load "ensemble" of expert agents
    expert_model_paths = [os.path.join(checkpoints_path, args.algorithm, model, "best_model.zip") for model in os.listdir(os.path.join(checkpoints_path, args.algorithm) if args.environment in model]
    #expert_model = (PPO if args.algorithm == "ppo" else SAC).load(
    #    os.path.join(checkpoints_path, args.algorithm, f"{args.environment.replace("/", "-")}_1", "best_model.zip")
    #)
    expert_models = []
    for model in expert_model_paths:
        expert_models.append((PPO if args.algorithm == "ppo" else SAC).load(
            os.path.join(checkpoints_path, args.algorithm, f"{args.environment.replace("/", "-")}_1", "best_model.zip")
        )

    if "procgen" in args.environment:
        _, short_name, _ = args.environment.split("-")
        environment = Gym3ToGymnasium(ProcgenGym3Env(num=1, env_name=short_name))
        environment = SaveResetEnvWrapper(TransformObservation(environment, lambda obs: obs["rgb"], environment.observation_space))
    elif "ALE/" in args.environment:
        environment = FrameStackObservation(AtariWrapper(gym.make(args.environment)), 4)
        environment = SaveResetEnvWrapper(TransformObservation(environment, lambda obs: obs.squeeze(-1), environment.observation_space))
    elif "MiniGrid" in args.environment:
        environment = SaveResetEnvWrapper(FlatObsWrapper(gym.make(args.environment)))
    else:
        environment = SaveResetEnvWrapper(gym.make(args.environment))
    
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

    feedback_path.parent.mkdir(parents=True, exist_ok=True)
    with open(feedback_path, "wb") as feedback_file:
        pickle.dump(feedback, feedback_file, protocol=pickle.HIGHEST_PROTOCOL)

if __name__ == "__main__":
    main()
