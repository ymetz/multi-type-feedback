#!/bin/bash

# Set the experiment parameters
# envs=("Ant-v5" "Humanoid-v5" "HalfCheetah-v5" "MiniGrid-GoToDoor-5x5-v0" "procgen-coinrun-v0" "procgen-miner-v0" "procgen-maze-v0" "ALE/Pong-v5" "ALE/MsPacman-v5")
envs=("ALE/Pong-v5" "ALE/MsPacman-v5")
cuda_devices=(1 1 1 2 2 3 3 4 4)

# Loop over the environments and CUDA devices
for i in "${!envs[@]}"; do
  export CUDA_VISIBLE_DEVICES=${cuda_devices[$i]}
  echo "Generate feedback for ${envs[$i]} with CUDA_VISIBLE_DEVICES=${cuda_devices[$i]}"
  python rlhf/generate_feedback_2.py --algorithm ppo --environment ${envs[$i]} --seed 1789 --n-feedback 1000 --save-folder feedback_debug &
done

# Wait for all training processes to finish
wait

echo "Feedback generated for all environments."
