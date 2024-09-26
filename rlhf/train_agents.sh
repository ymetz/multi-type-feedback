#!/bin/bash

# Set the experiment parameters
#envs=("Ant-v5" "Humanoid-v5" "HalfCheetah-v5" "MiniGrid-GoToDoor-5x5-v0" "procgen-coinrun-v0" "procgen-miner-v0" "procgen-maze-v0" "ALE/Pong-v5" "ALE/MsPacman-v5")
#envs=("Swimmer-v5" "HalfCheetah-v5" "Hopper-v5" "Walker2d-v5")
envs=("Ant-v5")
cuda_devices=(0 1 2 3 4 0 1 2 3 4 0 1 2 3 4)
seeds=(1789 1687123 12 912391 330)
#feedback_types=("evaluative" "comparative" "demonstrative" "corrective") #"descriptive" "descriptive_preference")
feedback_types=("descriptive" "cluster_description")

# Loop over the environments and CUDA devices
for seed in "${!seeds[@]}"; do
    for i in "${!envs[@]}"; do
      for j in "${!feedback_types[@]}"; do
          export CUDA_VISIBLE_DEVICES=${cuda_devices[$j]}
          echo "Train Agent for ${envs[$i]} and FB Type ${feedback_types[$j]} with CUDA_VISIBLE_DEVICES=${cuda_devices[$j]}"
              python rlhf/train_agent_2.py --algorithm ppo --environment ${envs[$i]} --feedback-type ${feedback_types[$j]} --seed ${seeds[$seed]} &
      done
    done
    # Wait for all training processes to finish
    #wait
done

echo "Feedback generated for all environments."
