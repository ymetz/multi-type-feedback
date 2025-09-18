#!/bin/bash

# ==============================
# Config
# ==============================
# Fixed total feedback budget
FEEDBACK_BUDGET=1500

# Number of iterations options for hyperparameter tuning
nr_of_iterations_opts=(10 20 50)

# Environments - mix of continuous control tasks
#envs=("highway-fast-v0" "merge-v0" "roundabout-v0")
envs=("metaworld-sweep-into-v3" "metaworld-pick-place-v3" "metaworld-button-press-v3")
seeds=(1789 12 912391)

# Feedback type combinations to test
#single_feedback=("evaluative" "comparative" "demonstrative" "corrective" "descriptive" "supervised")
single_feedback=("demonstrative" "corrective")

# Reward model type
reward_model_type="separate"

# Key hyperparameters
reward_training_epochs=50
feedback_buffer_sizes=(750)
sampling_strategy="random"

# Initial feedback count options
initial_feedback_count_opts=(250 500)

# (Unused here but left for completeness if you later extend)
shared_layers=5
ensemble_sizes=4

# Create directories
mkdir -p logs job_scripts results

# Function to create job script
create_job_script() {
    local job_id=$1
    local cmd=$2
    local time_limit=$3
    local job_name=$4
    
    cat <<EOT > job_scripts/job_${job_id}.sh
#!/bin/bash
#SBATCH --partition=cpu,cpu_il
#SBATCH --cpus-per-task=4
#SBATCH --ntasks=1
#SBATCH --job-name=${job_name}
#SBATCH --time=${time_limit}
#SBATCH --output=logs/${job_name}_%j.out
#SBATCH --error=logs/${job_name}_%j.err

source /pfs/data6/home/kn/kn_kn/kn_pop257914/ws_feedback_querying/venv/bin/activate

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

echo "Phase 1: Creating single feedback type baseline jobs with budget=${FEEDBACK_BUDGET}..."

# Loop seeds/envs/feedback types and hyperparams
for seed in "${seeds[@]}"; do
  for env in "${envs[@]}"; do
    for feedback in "${single_feedback[@]}"; do
      for init_feedback in "${initial_feedback_count_opts[@]}"; do
        for buffer_size in "${feedback_buffer_sizes[@]}"; do
          for nr_iterations in "${nr_of_iterations_opts[@]}"; do

            # Check if initial feedback exceeds budget
            if (( init_feedback >= FEEDBACK_BUDGET )); then
              echo "Skipping init=${init_feedback} (exceeds budget ${FEEDBACK_BUDGET})"
              continue
            fi

            # Build job name with key params visible
            job_name="baseline_${env}_${feedback}_s${seed}_init${init_feedback}_iter${nr_iterations}_budget${FEEDBACK_BUDGET}"

            # Command: now much simpler, let Python handle the budget calculations
            cmd="python multi_type_feedback/dynamic_rlhf.py \
                --algorithm ppo \
                --environment ${env} \
                --feedback-types ${feedback} \
                --reward-model-type ${reward_model_type} \
                --seed ${seed} \
                --expert-algorithm sac \
                --feedback-budget ${FEEDBACK_BUDGET} \
                --nr-of-iterations ${nr_iterations} \
                --reward-training-epochs ${reward_training_epochs} \
                --feedback-buffer-size ${buffer_size} \
                --initial-feedback-count ${init_feedback} \
                --sampling-strategy ${sampling_strategy} \
                --reference-data-folder ../multi-type-feedback_iclr2025/rlhf/feedback \
                --expert-model-base-path gt_agents \
                --wandb-project-name single_baselines_budget${FEEDBACK_BUDGET}"

            # Time limit heuristic: adjust based on your experience
            create_job_script $job_counter "$cmd" "03:00:00" $job_name
            ((job_counter++))
          done
        done
      done
    done
  done
done

# Create a submission script that submits jobs in batches
cat <<'EOT' > submit_jobs_baselines.sh
#!/bin/bash

# Configuration
MAX_CONCURRENT_JOBS=100
SLEEP_TIME=60
TOTAL_JOBS=$(ls job_scripts/job_*.sh 2>/dev/null | wc -l)

echo "Total jobs to submit: $TOTAL_JOBS"

# Function to get number of running/pending jobs
get_job_count() {
    squeue -u $USER | grep -E "(hp_sweep|baseline|arch|sweep|ensemble)" | wc -l
}

# Submit jobs in batches
job_id=0
while [ $job_id -lt $TOTAL_JOBS ]; do
    current_jobs=$(get_job_count)
    jobs_to_submit=$((MAX_CONCURRENT_JOBS - current_jobs))

    if [ $jobs_to_submit -gt 0 ]; then
        for ((i=0; i<jobs_to_submit && job_id<TOTAL_JOBS; i++)); do
            sbatch job_scripts/job_${job_id}.sh
            echo "Submitted job $job_id"
            ((job_id++))
            sleep 2
        done
    else
        echo "Queue full ($current_jobs jobs). Waiting..."
        sleep $SLEEP_TIME
    fi
done

echo "All jobs submitted!"
EOT

chmod +x submit_jobs_baselines.sh

echo "=================================="
echo "Sweep setup complete!"
echo "Total jobs created: $job_counter"
echo "=================================="
echo "Budget configuration:"
echo "  Total feedback budget: ${FEEDBACK_BUDGET}"
echo "  Number of iterations options: ${nr_of_iterations_opts[@]}"
echo "  Python script will automatically calculate:"
echo "    - n_feedback_per_iteration = (budget - initial_feedback) / nr_of_iterations"
echo "    - rl_steps_per_iteration = total_timesteps / nr_of_iterations"
echo "    - total_timesteps comes from ExperimentManager based on environment"
echo "=================================="
echo "To submit jobs, run: ./submit_jobs_baselines.sh"
echo "=================================="