#!/bin/bash
set -euo pipefail

# -------------------- Params --------------------
# envs=("Swimmer-v5" "HalfCheetah-v5" "Walker2d-v5")
# envs=("Ant-v5" "Hopper-v5" "Humanoid-v5")
# envs=("metaworld-button-press-v2" "metaworld-sweep-into-v2" "metaworld-pick-place-v2")
# envs=("roundabout-v0" "merge-v0" "highway-fast-v0")
envs=("Swimmer-v5" "HalfCheetah-v5" "Walker2d-v5" "merge-v0")

# seeds=(1789 1687123 12 912391 330)
seeds=(1789 1687123 12 912391 330)

# feedback_types=("evaluative" "comparative" "corrective" "descriptive" "descriptive_preference")
feedback_types=("comparative")

# noise_levels=(0.1 0.25 0.5 0.75 1.5 3.0)
# noise_levels=(0.0 0.1 0.25 0.5)
noise_levels=(0.0)

n_feedbacks=(5000) # default, use all
rt_loss_weights=(0.0 1.0)

# Only used when rt_loss_weight == 1.0
partitioners=("none" "random" "round_robin")
partition_sizes=(4 8 16 32)

# How many runs to pack into one sbatch (they will share a single GPU in this template)
batch_size=4

# -------------------- Prep --------------------
mkdir -p logs
declare -a combinations

# -------------------- Build combinations --------------------
for seed in "${seeds[@]}"; do
  for env in "${envs[@]}"; do
    for feedback in "${feedback_types[@]}"; do
      for noise in "${noise_levels[@]}"; do
        for n_feedback in "${n_feedbacks[@]}"; do
          for rt_loss_weight in "${rt_loss_weights[@]}"; do
            if [[ "${rt_loss_weight}" == "1.0" ]]; then
              # Expand across partitioners/sizes only for RT runs
              for partitioner in "${partitioners[@]}"; do
                for partition_size in "${partition_sizes[@]}"; do
                  combinations+=("$seed|$env|$feedback|$noise|$n_feedback|$rt_loss_weight|$partitioner|$partition_size")
                done
              done
            else
              # For non-RT, don't fan out (store neutral placeholders)
              combinations+=("$seed|$env|$feedback|$noise|$n_feedback|$rt_loss_weight|none|0")
            fi
          done
        done
      done
    done
  done
done

total_combinations=${#combinations[@]}

# -------------------- Submit in batches --------------------
for ((i=0; i<total_combinations; i+=batch_size)); do
  batch=("${combinations[@]:$i:$batch_size}")
  batch_id=$((i / batch_size))

  sbatch_script="batch_job_${batch_id}.sh"
  cat > "$sbatch_script" <<'EOT'
#!/bin/bash
#SBATCH --partition=gpu_h100,gpu_a100_il,gpu_h100_il,gpu_a100_short
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --ntasks=1
#SBATCH --job-name=train_reward_models_$SLURM_JOB_ID
#SBATCH --time=00:30:00
#SBATCH --output=logs/train_reward_models_%x_%j.out

module load devel/cuda/12.8
# source /pfs/data5/home/kn/kn_kn/kn_pop257914/multi-type-feedback/venv/bin/activate
EOT

  # Add each run; include partition flags only when rt_loss_weight==1.0
  for item in "${batch[@]}"; do
    IFS='|' read -r seed env feedback noise n_feedback rt_loss_weight partitioner partition_size <<< "$item"

    extra_flags=""
    if [[ "${rt_loss_weight}" == "1.0" ]]; then
      extra_flags+=" --partitioner ${partitioner} --partition-size ${partition_size}"
    fi

    cat >> "$sbatch_script" <<EOC
python multi_type_feedback/train_reward_model.py \\
  --algorithm ppo \\
  --environment "${env}" \\
  --feedback-type "${feedback}" \\
  --n-feedback "${n_feedback}" \\
  --seed "${seed}" \\
  --noise-level "${noise}" \\
  --rt-loss-weight "${rt_loss_weight}" \\
  --no-loading-bar \\
  --wandb-project-name ablations${extra_flags} &
EOC
  done

  echo "wait" >> "$sbatch_script"

  sbatch "$sbatch_script"
  rm "$sbatch_script"
done

echo "All jobs have been submitted."
