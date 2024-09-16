#!/bin/bash

# Set the experiment parameters
# envs=("Ant-v5" "Swimmer-v5" "HalfCheetah-v5" "Hopper-v5" "ALE/MsPacman-v5" "ALE/BeamRider-v5" "ALE/Enduro-v5")
# envs=("Ant-v5" "Swimmer-v5" "HalfCheetah-v5" "Hopper-v5")
envs=("Ant-v5" "Swimmer-v5" "HalfCheetah-v5" "Hopper-v5" "Walker2d-v5")
cuda_devices=(0 1 2 3 4 0 1 2 3 4 0 1 2 3 4)
seeds=(1789 1687123 12 912391 330)

for seed in "${!seeds[@]}"; do
    for i in "${!envs[@]}"; do
      export CUDA_VISIBLE_DEVICES=${cuda_devices[$i]}
      echo "Generate feedback for ${envs[$i]} with CUDA_VISIBLE_DEVICES=${cuda_devices[$i]}"
      python rlhf/generate_feedback_2.py --algorithm ppo --environment ${envs[$i]} --seed ${seeds[$seed]} --n-feedback 10000 --save-folder feedback &
    done
    
    # Wait for all training processes to finish
    wait
done

echo "Feedback generated for all environments."
