#!/bin/bash

# Environments
#envs=("Swimmer-v5" "Walker2d-v5" "HalfCheetah-v5" "highway-fast-v0" "merge-v0" "roundabout-v0")
envs=("metaworld-sweep-into-v3" "metaworld-pick-place-v3" "metaworld-button-press-v3")
seeds=(1789 12 912391)

# Multi-feedback combo(s)
combo_feedback=(
    "evaluative comparative demonstrative corrective descriptive"
)

# Reward model architectures
reward_model_types=("separate" "multi-head" "unified")

# Core knobs (updated)
nr_of_iterations=(10 20 40)
reward_training_epochs=(30)
feedback_buffer_size=750
sampling_strategies=("random" "uncertainty")

# Fixed network settings (no layer sweeps)
shared_layer_number=5
head_layer_num=1

# Ensemble sizes (Phase 4)
ensemble_sizes=(4)

# Create directories
mkdir -p logs
mkdir -p job_scripts_big_sweep
mkdir -p results

# Function to create job script
create_job_script() {
    local job_id=$1
    local cmd=$2
    local time_limit=$3
    local job_name=$4
    
    cat <<EOT > job_scripts_big_sweep/job_${job_id}.sh
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

###############################################
# Phase 2: Multi-feedback architecture comps  #
###############################################
# (No single-type baselines; fixed net layers; use --nr-of-iterations)
echo "Phase 2: Creating multi-feedback architecture comparison jobs..."
for seed in "${seeds[@]}"; do
    for env in "${envs[@]}"; do
        for feedback_combo in "${combo_feedback[@]}"; do
            for model_type in "${reward_model_types[@]}"; do
                job_name="arch_${env}_${model_type}_s${seed}_$(echo $feedback_combo | sed 's/ /_/g')"
                cmd="python multi_type_feedback/dynamic_rlhf.py \
                    --algorithm ppo \
                    --environment $env \
                    --feedback-types $feedback_combo \
                    --reward-model-type $model_type \
                    --seed $seed \
                    --expert-algorithm sac \
                    --nr-of-iterations 20 \
                    --reward-training-epochs 20 \
                    --feedback-buffer-size 750 \
                    --sampling-strategy random \
                    --shared-layer-number $shared_layer_number \
                    --head-layer-num $head_layer_num \
                    --reference-data-folder ../multi-type-feedback_iclr2025/rlhf/feedback \
                    --expert-model-base-path gt_agents \
                    --wandb-project-name dynamic_rlhf_joint_sweep"
                create_job_script $job_counter "$cmd" "05:00:00" $job_name
                ((job_counter++))
            done
        done
    done
done

######################################################
# Phase 3: Hyperparameter sweep on single environment #
######################################################
echo "Phase 3: Creating hyperparameter sweep jobs..."

for seed in "${seeds[@]}"; do
    for iters in "${nr_of_iterations[@]}"; do
        for epochs in "${reward_training_epochs[@]}"; do
            for strategy in "${sampling_strategies[@]}"; do
                for env in "${envs[@]}"; do
                    job_name="sweep_${best_env}_it${iters}_ep${epochs}_buf${buffer_size}_rl${rl_steps}_${strategy}_s${seed}"
                    cmd="python multi_type_feedback/dynamic_rlhf.py \
                        --algorithm ppo \
                        --environment $env \
                        --feedback-types evaluative comparative demonstrative corrective descriptive \
                        --reward-model-type unified \
                        --seed $seed \
                        --expert-algorithm sac \
                        --nr-of-iterations $iters \
                        --reward-training-epochs $epochs \
                        --feedback-buffer-size 750 \
                        --sampling-strategy $strategy \
                        --shared-layer-number $shared_layer_number \
                        --head-layer-num $head_layer_num \
                        --reference-data-folder ../multi-type-feedback_iclr2025/rlhf/feedback \
                        --expert-model-base-path gt_agents \
                        --wandb-project-name dynamic_rlhf_joint_sweep"
                    create_job_script $job_counter "$cmd" "05:00:00" $job_name
                    ((job_counter++))
                done
            done
        done
    done
done

# Create a submission script that submits jobs in batches
cat <<EOT > submit_jobs_sweep.sh
#!/bin/bash

# Configuration
MAX_CONCURRENT_JOBS=50
SLEEP_TIME=60
TOTAL_JOBS=${job_counter}

echo "Total jobs to submit: \$TOTAL_JOBS"

# Function to get number of running/pending jobs
get_job_count() {
    squeue -u \$USER | grep -E "(arch|sweep|ensemble)" | wc -l
}

# Submit jobs in batches
job_id=0
while [ \$job_id -lt \$TOTAL_JOBS ]; do
    current_jobs=\$(get_job_count)
    jobs_to_submit=\$((MAX_CONCURRENT_JOBS - current_jobs))
    if [ \$jobs_to_submit -gt 0 ]; then
        for ((i=0; i<jobs_to_submit && job_id<TOTAL_JOBS; i++)); do
            sbatch job_scripts_big_sweep/job_\${job_id}.sh
            echo "Submitted job \$job_id"
            ((job_id++))
            sleep 2
        done
    else
        echo "Queue full (\$current_jobs jobs). Waiting..."
        sleep \$SLEEP_TIME
    fi
done

echo "All jobs submitted!"
EOT

chmod +x submit_jobs_sweep.sh

echo "=================================="
echo "Sweep setup complete!"
echo "Total jobs created: $job_counter"
echo "=================================="
echo "Job distribution:"
echo "- Phase 2 (Architectures; fixed layers): variable"
echo "- Phase 3 (Hyperparameter sweep on ${best_env} with nr_of_iterations): variable"
echo "=================================="
echo "To submit jobs, run: ./submit_jobs_sweep.sh"
echo "=================================="
