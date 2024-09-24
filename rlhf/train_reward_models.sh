#!/bin/bash

# Set the experiment parameters
envs=("Ant-v5" "Swimmer-v5" "HalfCheetah-v5" "Hopper-v5" "Walker2d-v5")
cuda_devices=(0 1 2 3 4 0 1 2 3 4 0 1 2 3 4)
#seeds=(1789 1687123 12 912391 330)
seeds=(1789 1687123 12 912391 330)
feedback_types=("evaluative" "comparative" "demonstrative" "corrective") #"descriptive" "descriptive_preference")

# Loop over the environments and CUDA devices
for seed in "${!seeds[@]}"; do
    for i in "${!envs[@]}"; do
        for j in "${!feedback_types[@]}"; do
          export CUDA_VISIBLE_DEVICES=${cuda_devices[$j]}
          echo "Train Reward Models for ${envs[$i]} and FB Type ${feedback_types[$j]} with CUDA_VISIBLE_DEVICES=${cuda_devices[$j]} and SEED ${seeds[$seed]}"
          python rlhf/train_reward_model_2.py --algorithm ppo --environment ${envs[$i]} --feedback-type ${feedback_types[$j]} --seed ${seeds[$seed]} &
        done

        # Wait for all training processes to finish
        wait

    done
  
done

echo "Feedback generated for all environments."
