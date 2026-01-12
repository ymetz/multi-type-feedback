#!/bin/bash

envs=("Swimmer-v5" "HalfCheetah-v5" "Walker2d-v5""merge-v0" ) # sweep over all envs

seeds=(1789 1687123 12 912391 330) # we use five seeds
save_freqs=(50000 50000 50000 50000) # we try to collect 20 checkpoints for diversity, so this is total_timesteps // 20

# Create a directory for log files if it doesn't exist
mkdir -p logs

# Prepare an array to hold all combinations
declare -a combinations

# Generate combinations with matched save frequencies
for seed in "${seeds[@]}"; do
    for i in "${!envs[@]}"; do
        combinations+=("$seed ${envs[$i]} ${save_freqs[$i]}")
    done
done

batch_size=1
total_combinations=${#combinations[@]}

# Loop over the combinations in batches
for ((i=0; i<$total_combinations; i+=$batch_size)); do
    batch=("${combinations[@]:$i:$batch_size}")
    batch_id=$((i / batch_size))
    
    # Create a temporary Slurm job script for this batch
    sbatch_script="batch_job_$batch_id.sh"
    
    cat <<EOT > $sbatch_script
#!/bin/bash
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=8
#SBATCH --ntasks=1
#SBATCH --job-name=train_experts_$batch_id
#SBATCH --time=02:00:00
#SBATCH --output=logs/train_experts_${batch_id}_%j.out

# Load any necessary modules or activate environments here
# source /venv/bin/activate

# Run the training jobs in background
EOT
    
    # Add each task to the Slurm script
    for combination in "${batch[@]}"; do
        read seed env save_freq <<< "$combination"
        echo "python train_baselines/train.py --algo ppo --env $env --verbose 0 --save-freq $save_freq --seed $seed --log-folder gt_agents &" >> $sbatch_script
    done
    
    # Wait for all background jobs to finish
    echo "wait" >> $sbatch_script
    
    # Submit the Slurm job script
    echo "Submitting batch $batch_id..."
    sbatch $sbatch_script
    
    # Remove the temporary Slurm script
    rm $sbatch_script
    
    # Add delay between job submissions (except for the last job) - avoids race condition for saving
    if [ $((i + batch_size)) -lt $total_combinations ]; then
        echo "Waiting 20 seconds before submitting next batch..."
        sleep 20
    fi
done

echo "All jobs have been submitted."