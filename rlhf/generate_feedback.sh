#!/bin/bash

# Set the experiment parameters
envs=("Ant-v5" "Swimmer-v5" "HalfCheetah-v5" "Hopper-v5" "procgen-coinrun-v0" "procgen-miner-v0" "ALE/MsPacman-v5" "ALE/BeamRider-v5" "ALE/Enduro-v5" "ALE/Pong-v5")
cuda_devices=(0 0 0 1 1 2 2 3 3 4 4)

# Loop over the environments and CUDA devices
for i in "${!envs[@]}"; do
  export CUDA_VISIBLE_DEVICES=${cuda_devices[$i]}
  echo "Generate feedback for ${envs[$i]} with CUDA_VISIBLE_DEVICES=${cuda_devices[$i]}"
  python rlhf/generate_feedback_2.py --algorithm ppo --environment ${envs[$i]} --seed 1789 --n-feedback 10000 --save-folder feedback &
done

# Wait for all training processes to finish
wait

echo "Feedback generated for all environments."
