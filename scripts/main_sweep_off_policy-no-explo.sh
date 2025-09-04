#!/bin/bash

# Environments - mix of continuous control tasks with different characteristics
envs=("Swimmer-v5" "Walker2d-v5" "HalfCheetah-v5")  # Added HalfCheetah for diversity
seeds=(1789 12 912391 330 1687123)  # Reduced to 3 for faster initial sweep

# Feedback type combinations to test
# Single feedback types
feedback_types=("evaluative" "comparative" "demonstrative" "corrective" "descriptive" "supervised")

query_strategies=("none" "average" "min" "max")

reward_model_types=("separate" "multi-head" "unified")

# Create directories
mkdir -p logs
mkdir -p job_scripts_off_policy
mkdir -p results

# Function to create job script
create_job_script() {
    local job_id=$1
    local cmd=$2
    local time_limit=$3
    local job_name=$4
    
    cat <<EOT > job_scripts_off_policy/job_${job_id}.sh
#!/bin/bash
#SBATCH --partition=cpu,cpu_il
#SBATCH --cpus-per-task=4
#SBATCH --ntasks=1
#SBATCH --job-name=${job_name}
#SBATCH --time=${time_limit}
#SBATCH --output=logs/${job_name}_%j.out

# Load environment
source /pfs/data5/home/kn/kn_kn/kn_pop257914/ws_feedback_querying/venv/bin/activate

# Force CPU usage
export CUDA_VISIBLE_DEVICES=""
export OMP_NUM_THREADS=4

# Run the job
$cmd

# Save exit code
echo "Job finished with exit code: \$?" >> logs/${job_name}_status.txt
EOT
}

job_counter=0

# first submit single types
for seed in "${seeds[@]}"; do
    for env in "${envs[@]}"; do
        for feedback in "${feedback_types[@]}"; do
            for query_strategy in "${query_strategies[@]}"; do
                # Use default hyperparameters for baseline
                job_name="baseline_${env}_${feedback}_s${seed}_q${query_strategy}"
                
                cmd="python multi_type_feedback/dynamic_rlhf.py \
                    --algorithm sac \
                    --environment $env \
                    --feedback-types $feedback \
                    --reward-model-type separate \
                    --seed $seed \
                    --n-feedback-per-iteration 25 \
                    --reward-training-epochs 20 \
                    --feedback-buffer-size 2000 \
                    --initial-feedback-count 500 \
                    --rl-steps-per-iteration 50000 \
                    --query-sampling-strategy $query_strategy \
                    --sampling-strategy uncertainty \
                    --reference-data-folder feedback \
                    --expert-model-base-path gt_agents \
                    --wandb-project-name dynamic_rlhf_main_sweep_sac"
                
                create_job_script $job_counter "$cmd" "04:30:00" $job_name
                ((job_counter++))
            done
        done
    done
done

# MTF-PEBBLE
for seed in "${seeds[@]}"; do
    for env in "${envs[@]}"; do
        for query_strategy in "${query_strategies[@]}"; do
            for reward_model_type in "${reward_model_types[@]}"; do 
                # Use default hyperparameters for baseline
                job_name="mtf-pebble_${env}_s${seed}_q${query_strategy}_r${reward_model_type}"
                
                cmd="python multi_type_feedback/dynamic_rlhf.py \
                    --algorithm sac \
                    --environment $env \
                    --feedback-types evaluative comparative demonstrative corrective descriptive \
                    --reward-model-type $reward_model_type \
                    --seed $seed \
                    --n-feedback-per-iteration 25 \
                    --reward-training-epochs 20 \
                    --feedback-buffer-size 2000 \
                    --initial-feedback-count 500 \
                    --rl-steps-per-iteration 50000 \
                    --query-sampling-strategy $query_strategy \
                    --sampling-strategy uncertainty \
                    --reference-data-folder feedback \
                    --expert-model-base-path gt_agents \
                    --wandb-project-name dynamic_rlhf_main_sweep_sac"
                
                create_job_script $job_counter "$cmd" "04:30:00" $job_name
                ((job_counter++))
            done
        done
    done
done


# Create a submission script that submits jobs in batches
cat <<EOT > submit_jobs_off_policy.sh
#!/bin/bash

# Configuration
MAX_CONCURRENT_JOBS=140
SLEEP_TIME=60
TOTAL_JOBS=${job_counter}

echo "Total jobs to submit: \$TOTAL_JOBS"

# Function to get number of running/pending jobs
get_job_count() {
    squeue -u \$USER | grep -E "(hp_sweep|baseline|arch|sweep|ensemble)" | wc -l
}

# Submit jobs in batches
job_id=0
while [ \$job_id -lt \$TOTAL_JOBS ]; do
    current_jobs=\$(get_job_count)
    
    # Calculate how many jobs we can submit
    jobs_to_submit=\$((MAX_CONCURRENT_JOBS - current_jobs))
    
    if [ \$jobs_to_submit -gt 0 ]; then
        # Submit up to jobs_to_submit jobs
        for ((i=0; i<jobs_to_submit && job_id<TOTAL_JOBS; i++)); do
            sbatch job_scripts_off_policy/job_\${job_id}.sh
            echo "Submitted job \$job_id"
            ((job_id++))
            sleep 2  # Small delay between submissions
        done
    else
        echo "Queue full (\$current_jobs jobs). Waiting..."
        sleep \$SLEEP_TIME
    fi
done

echo "All jobs submitted!"
EOT

chmod +x submit_jobs_off_policy.sh

# Create analysis script
cat <<'EOT' > analyze_results.py
#!/usr/bin/env python3
import pandas as pd
import wandb
import numpy as np
from collections import defaultdict

def analyze_sweep_results(project_name="dynamic_rlhf_big_sweep"):
    """Analyze results from hyperparameter sweep."""
    api = wandb.Api()
    runs = api.runs(f"{wandb.Api().default_entity}/{project_name}")
    
    results = defaultdict(list)
    
    for run in runs:
        if run.state != "finished":
            continue
            
        config = run.config
        summary = run.summary
        
        # Extract key metrics
        results['environment'].append(config.get('environment'))
        results['feedback_types'].append(config.get('feedback_types'))
        results['reward_model_type'].append(config.get('reward_model_type'))
        results['n_feedback_per_iteration'].append(config.get('n_feedback_per_iteration'))
        results['reward_training_epochs'].append(config.get('reward_training_epochs'))
        results['feedback_buffer_size'].append(config.get('feedback_buffer_size'))
        results['sampling_strategy'].append(config.get('sampling_strategy'))
        results['final_reward'].append(summary.get('rollout/ep_rew_mean', np.nan))
        results['total_timesteps'].append(summary.get('time/total_timesteps', 0))
        
    df = pd.DataFrame(results)
    
    # Analysis
    print("=== Best Configurations by Environment ===")
    for env in df['environment'].unique():
        env_df = df[df['environment'] == env]
        best_idx = env_df['final_reward'].idxmax()
        if not pd.isna(best_idx):
            print(f"\n{env}:")
            print(env_df.loc[best_idx])
    
    # Aggregate statistics
    print("\n=== Feedback Type Performance ===")
    feedback_perf = df.groupby('feedback_types')['final_reward'].agg(['mean', 'std', 'count'])
    print(feedback_perf.sort_values('mean', ascending=False))
    
    # Save detailed results
    df.to_csv('sweep_results.csv', index=False)
    print("\nDetailed results saved to sweep_results.csv")

if __name__ == "__main__":
    analyze_sweep_results()
EOT

chmod +x analyze_results.py

echo "=================================="
echo "Sweep setup complete!"
echo "Total jobs created: $job_counter"
echo "=================================="
echo "Job distribution:"
echo "- Phase 1 (Baselines): $((${#seeds[@]} * ${#envs[@]} * ${#single_feedback[@]}))"
echo "- Phase 2 (Architectures): Variable based on combinations"
echo "=================================="
echo "To submit jobs, run: ./submit_jobs_off_policy.sh"
echo "To analyze results after completion: ./analyze_results.py"
echo "=================================="