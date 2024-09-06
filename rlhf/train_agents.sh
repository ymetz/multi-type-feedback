#!/bin/bash

# Set the experiment parameters
envs=("Ant-v5" "Humanoid-v5" "HalfCheetah-v5" "MiniGrid-GoToDoor-5x5-v0" "procgen-coinrun-v0" "procgen-miner-v0" "procgen-maze-v0" "ALE/Pong-v5" "ALE/MsPacman-v5")
cuda_devices=(1 1 2 2 3 4)
feedback_types=("evaluative" "comparative" "demonstrative" "corrective" "descriptive" "descriptive_preference")

# Loop over the environments and CUDA devices
for i in "${!envs[@]}"; do
  for j in "${!feedback_types[@]}"; do
      export CUDA_VISIBLE_DEVICES=${cuda_devices[$j]}
      echo "Train Agent for ${envs[$i]} and FB Type ${feedback_types[$j]} with CUDA_VISIBLE_DEVICES=${cuda_devices[$j]}"
      python rlhf/train_agent_2.py --algorithm ppo --environment ${envs[$i]} --feedback-type ${feedback_types[$j]} &
  done

  # Wait for all training processes to finish
  wait
  
done

echo "Feedback generated for all environments."
