#!/bin/bash

# ==============================
# Config
# ==============================
# Fixed total timesteps & feedback budget
TOTAL_TIMESTEPS=1000000
FEEDBACK_BUDGET=1500

# Environments - mix of continuous control tasks
envs=("Swimmer-v5" "Walker2d-v5" "HalfCheetah-v5")
seeds=(1789 12 912391)

# Feedback type combinations to test
single_feedback=("evaluative" "comparative" "demonstrative" "corrective" "descriptive" "supervised")

# Reward model type
reward_model_type="separate"

# Key hyperparameters (will plug into the call just like your sweep)
reward_training_epochs=50
feedback_buffer_size=5000
sampling_strategy="random"

# We'll *compute* n_feedback_per_iteration from the budget for these choices:
initial_feedback_count_opts=(250 500)
rl_steps_per_iteration_opts=(10000 20000 40000 100000)

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

echo "Phase 1: Creating single feedback type baseline jobs with fixed budget=${FEEDBACK_BUDGET}..."

# Loop seeds/envs/feedback types and hyperparams where we enforce the budget
for seed in "${seeds[@]}"; do
  for env in "${envs[@]}"; do
    for feedback in "${single_feedback[@]}"; do
      for init_feedback in "${initial_feedback_count_opts[@]}"; do
        for rl_steps in "${rl_steps_per_iteration_opts[@]}"; do

          # Number of RL updates
          updates=$(( TOTAL_TIMESTEPS / rl_steps ))

          # If TOTAL_TIMESTEPS not divisible by rl_steps, skip (shouldn't happen with given opts)
          if (( updates * rl_steps != TOTAL_TIMESTEPS )); then
            echo "Skipping rl_steps=${rl_steps} (not dividing TOTAL_TIMESTEPS=${TOTAL_TIMESTEPS})"
            continue
          fi

          # Remaining budget after initial feedback
          remaining=$(( FEEDBACK_BUDGET - init_feedback ))
          if (( remaining <= 0 )); then
            echo "Skipping init=${init_feedback} (exceeds or equals budget)"
            continue
          fi

          # Check if an integer n_feedback_per_iteration exists
          if (( remaining % updates != 0 )); then
            # Not an integer; skip this combo
            continue
          fi

          n_feedback=$(( remaining / updates ))
          if (( n_feedback <= 0 )); then
            continue
          fi

          # Build job name with key params visible
          job_name="baseline_${env}_${feedback}_s${seed}_init${init_feedback}_rl${rl_steps}_nf${n_feedback}"

          # Command: now includes all the “extra” args (like your sweep), but with computed n_feedback
          cmd="python multi_type_feedback/dynamic_rlhf.py \
              --algorithm ppo \
              --environment ${env} \
              --feedback-types ${feedback} \
              --reward-model-type ${reward_model_type} \
              --seed ${seed} \
              --n-feedback-per-iteration ${n_feedback} \
              --reward-training-epochs ${reward_training_epochs} \
              --feedback-buffer-size ${feedback_buffer_size} \
              --initial-feedback-count ${init_feedback} \
              --rl-steps-per-iteration ${rl_steps} \
              --sampling-strategy ${sampling_strategy} \
              --reference-data-folder ../multi-type-feedback_iclr2025/rlhf/feedback \
              --expert-model-base-path gt_agents \
              --wandb-project-name single_baselines_budget${FEEDBACK_BUDGET}"

          # Time limit heuristic: more RL steps => fewer updates, but same total timesteps;
          # keep a safe default; adjust if you have runtime stats
          create_job_script $job_counter "$cmd" "02:00:00" $job_name
          ((job_counter++))

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
echo "Budget logic:"
echo "  initial_feedback + (TOTAL_TIMESTEPS / rl_steps_per_iteration) * n_feedback_per_iteration = ${FEEDBACK_BUDGET}"
echo "  TOTAL_TIMESTEPS = ${TOTAL_TIMESTEPS}"
echo "=================================="
echo "To submit jobs, run: ./submit_jobs_baselines.sh"
echo "=================================="
