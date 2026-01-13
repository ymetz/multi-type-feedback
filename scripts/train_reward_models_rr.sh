#!/bin/bash

# --- Params ---
envs=("Swimmer-v5" "HalfCheetah-v5" "Walker2d-v5""merge-v0" ) # sweep over all envs

seeds=(1789 1687123 12 912391 330) # we use five seeds

noise_levels=(0.0 0.1 0.25 0.5) # different noise levels - use 0.0 for no noise

n_feedbacks=(5000) # default, use all
rr_loss_weights=(0.0 1.0) # two configurations: BT baseline and full ResponseRank

# Create a directory for log files if it doesn't exist
mkdir -p logs

# Prepare an array to hold all combinations
declare -a combinations

# Generate all combinations
for seed in "${seeds[@]}"; do
    for env in "${envs[@]}"; do
        for noise in "${noise_levels[@]}"; do
            for n_feedback in "${n_feedbacks[@]}"; do
                for rr_loss_weight in "${rr_loss_weights[@]}"; do
                    combinations+=("$seed $env $noise $n_feedback $rr_loss_weight")
                done
            done
        done
    done
done

# Set the batch size (number of jobs per GPU)
batch_size=4
total_combinations=${#combinations[@]}

# Loop over the combinations in batches
for ((i=0; i<$total_combinations; i+=$batch_size)); do
    batch=("${combinations[@]:$i:$batch_size}")
    batch_id=$((i / batch_size))

    # Create a temporary Slurm job script for this batch
    sbatch_script="batch_job_$batch_id.sh"
    cat <<EOT > $sbatch_script
#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --ntasks=1
#SBATCH --job-name=train_reward_models_$batch_id
#SBATCH --time=00:30:00
#SBATCH --output=logs/train_reward_models_${batch_id}_%j.out

# Load any necessary modules or activate environments here
module load devel/cuda/12.8

# Run the training jobs in background
EOT

    # Add each task to the Slurm script
    for combination in "${batch[@]}"; do
        read seed env feedback noise n_feedback rr_loss_weight <<< $combination
        echo "python multi_type_feedback/train_reward_model.py --algorithm ppo --environment $env --feedback-type comparative --n-feedback $n_feedback --seed $seed --noise-level $noise --rr-loss-weight $rr_loss_weight --no-loading-bar --wandb-project-name response_rank &" >> $sbatch_script
    done

    # Wait for all background jobs to finish
    echo "wait" >> $sbatch_script

    # Submit the Slurm job script
    sbatch $sbatch_script

    # Optional: Remove the temporary Slurm script
    rm $sbatch_script
done

echo "All jobs have been submitted."