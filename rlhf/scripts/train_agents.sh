#!/bin/bash

#envs=("HalfCheetah-v5" "Walker2d-v5" "Swimmer-v5")
#envs=("Ant-v5" "Hopper-v5" "Humanoid-v5")
#envs=("metaworld-pick-place-v2" "metaworld-button-press-v2")
#envs=("metaworld-sweep-into-v2" "metaworld-pick-place-v2" "metaworld-button-press-v2")
#envs=("highway-fast-v0" "roundabout-v0")    # ("merge-v0" "highway-fast-v0" "roundabout-v0")
#envs=("highway-fast-v0")
envs=("Humanoid-v5")
#seeds=(1789 1687123 12 912391 330)
seeds=(1789 1687123 12)
#feedback_types=("evaluative" "comparative" "demonstrative" "corrective" "descriptive" "descriptive_preference")
feedback_types=("evaluative" "comparative" "corrective" "demonstrative" "descriptive" "descriptive_preference")
#feedback_types=("corrective")
noise_levels=(0.1 0.25 0.5 0.75)
n_feedbacks=(-1)
#n_feedbacks=(5000 2500 1250 750)


# Create a directory for log files if it doesn't exist
mkdir -p logs

# Prepare an array to hold all combinations
declare -a combinations

# Generate all combinations
for seed in "${seeds[@]}"; do
    for env in "${envs[@]}"; do
        for feedback in "${feedback_types[@]}"; do
            for noise in "${noise_levels[@]}"; do
                for n_feedback in "${n_feedbacks[@]}"; do
                    combinations+=("$seed $env $feedback $noise $n_feedback")
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
#SBATCH --partition=gpu_4,gpu_8,gpu_4_a100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --ntasks=1
#SBATCH --job-name=train_agents_$batch_id
#SBATCH --time=08:00:00
#SBATCH --output=logs/train_agents_${batch_id}_%j.out

# Load any necessary modules or activate environments here
source /pfs/data5/home/kn/kn_kn/kn_pop257914/multi-type-feedback/venv/bin/activate

# Run the training jobs in background
EOT

    # Add each task to the Slurm script
    for combination in "${batch[@]}"; do
        read seed env feedback noise n_feedback <<< $combination
        echo "python rlhf/train_RL_agent.py --algorithm sac --environment $env --feedback-type $feedback --seed $seed --n-feedback $n_feedback --noise-level $noise &" >> $sbatch_script
    done

    # Wait for all background jobs to finish
    echo "wait" >> $sbatch_script

    # Submit the Slurm job script
    sbatch $sbatch_script

    # Optional: Remove the temporary Slurm script
    rm $sbatch_script
done

echo "All jobs have been submitted."