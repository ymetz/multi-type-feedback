#!/bin/bash

# Single environment for faster sweeping
envs=("LunarLander-v3")
seeds=(1789 1687123) #12 912391 330)
# Basic feedback types for initial sweep
feedback_types=("evaluative comparative demonstrative descriptive")
# Hyperparameter ranges
n_feedback_per_iteration=(10 30 50)
reward_training_epochs=(1 3 5 8)
feedback_buffer_size=(500 1000 2000 3000)

# Create a directory for log files and scripts if they don't exist
mkdir -p logs
mkdir -p job_scripts

# Counter for job scripts
job_counter=0

# Generate all combinations and create individual job scripts
for seed in "${seeds[@]}"; do
    for env in "${envs[@]}"; do
        for feedback in "${feedback_types[@]}"; do
            for n_feedback in "${n_feedback_per_iteration[@]}"; do
                for epochs in "${reward_training_epochs[@]}"; do
                    for buffer_size in "${feedback_buffer_size[@]}"; do
                        # Create a unique job script for this combination
                        job_script="job_scripts/job_${job_counter}.sh"
                        
                        # Create the job script with SLURM directives
                        cat <<EOT > $job_script
#!/bin/bash
#SBATCH --partition=single
#SBATCH --cpus-per-task=2
#SBATCH --ntasks=1
#SBATCH --job-name=hp_sweep_${job_counter}
#SBATCH --time=00:30:00
#SBATCH --output=logs/hp_sweep_${job_counter}_%j.out

# Load necessary modules or activate environments
source /pfs/data5/home/kn/kn_kn/kn_pop257914/ws_feedback_querying/venv/bin/activate

# Set environment variable to force CPU usage
export CUDA_VISIBLE_DEVICES=""

# Run the training job
python multi_type_feedback/dynamic_rlhf.py \
    --algorithm ppo \
    --environment $env \
    --feedback-types $feedback \
    --seed $seed \
    --n-feedback-per-iteration $n_feedback \
    --reward-training-epochs $epochs \
    --feedback-buffer-size $buffer_size \
    --wandb-project-name dynamic_rlhf_sweep
EOT

                        # Make the job script executable
                        chmod +x $job_script
                        
                        # Submit the job
                        sbatch $job_script
                        
                        # Increment the counter
                        ((job_counter++))
                    done
                done
            done
        done
    done
done

echo "All $job_counter jobs have been submitted."