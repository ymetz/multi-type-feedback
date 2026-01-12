#!/bin/bash
set -euo pipefail

# --- Params ---
# envs=("Swimmer-v5" "HalfCheetah-v5" "Walker2d-v5")
# envs=("Ant-v5" "Hopper-v5" "Humanoid-v5")
# envs=("metaworld-button-press-v2" "metaworld-sweep-into-v2" "metaworld-pick-place-v2")
# envs=("roundabout-v0" "merge-v0" "highway-fast-v0")
envs=("Walker2d-v5")

# seeds=(1789 1687123 12 912391 330)
seeds=(912391)

# feedback_types=("evaluative" "comparative" "corrective" "descriptive" "descriptive_preference")
feedback_types=("comparative")

# noise_levels=(0.0 0.1 0.25 0.5 0.75 1.5 3.0)
noise_levels=(0.5)

n_feedbacks=(5000) # default, use all
rt_loss_weights=(0.0)

# Fanout control
batch_size=1

# --- Prep ---
mkdir -p logs
declare -a combinations

# --- Build combinations ---
for seed in "${seeds[@]}"; do
  for env in "${envs[@]}"; do
    for feedback in "${feedback_types[@]}"; do
      for noise in "${noise_levels[@]}"; do
        for n_feedback in "${n_feedbacks[@]}"; do
          for rt_loss_weight in "${rt_loss_weights[@]}"; do
            if [[ "${rt_loss_weight}" == "1.0" ]]; then
              combinations+=("$seed|$env|$feedback|$noise|$n_feedback|$rt_loss_weight")
            else
              # No extra runs for non-RT; pass a neutral partitioner/size
              # (adjust 'none' and '0' if your script expects different sentinels)
              combinations+=("$seed|$env|$feedback|$noise|$n_feedback|$rt_loss_weight")
            fi
          done
        done
      done
    done
  done
done

total_combinations=${#combinations[@]}

# --- Submit in batches ---
for ((i=0; i<total_combinations; i+=batch_size)); do
  batch=("${combinations[@]:$i:$batch_size}")
  batch_id=$((i / batch_size))
  local_ntasks=${#batch[@]}

  sbatch_script="batch_job_${batch_id}.sh"
  cat > "$sbatch_script" <<EOT
#!/bin/bash
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=4
#SBATCH --ntasks=${local_ntasks}
#SBATCH --job-name=train_RL_agent_rt_${batch_id}
#SBATCH --time=05:30:00
#SBATCH --output=logs/train_RL_agent_${batch_id}_%j.out

# module load devel/cuda/12.8
# source /pfs/data5/home/kn/kn_kn/kn_pop257914/multi-type-feedback/venv/bin/activate

EOT

  for item in "${batch[@]}"; do
    IFS='|' read -r seed env feedback noise n_feedback rt_loss_weight <<< "$item"
    echo "python multi_type_feedback/train_RL_agent.py \\
      --algorithm ppo \\
      --environment \"$env\" \\
      --feedback-type \"$feedback\" \\
      --n-feedback \"$n_feedback\" \\
      --seed \"$seed\" \\
      --noise-level \"$noise\" \\
      --rt-loss-weight \"$rt_loss_weight\" \\
      --wandb-project-name rt_rank &" >> "$sbatch_script"
  done

  echo "wait" >> "$sbatch_script"
  sbatch "$sbatch_script"
  rm "$sbatch_script"
done

echo "All jobs have been submitted."
