#!/bin/bash
set -euo pipefail

# --- Params ---
envs=("Swimmer-v5" "HalfCheetah-v5" "Walker2d-v5""merge-v0" ) # sweep over all envs

seeds=(1789 1687123 12 912391 330) # we use five seeds

noise_levels=(0.0 0.1 0.25 0.5) # different noise levels - use 0.0 for no noise

n_feedbacks=(5000) # default, use all
rr_loss_weights=(0.0 1.0) # two configurations: BT baseline and full ResponseRank

# Fanout control (how many runs per batch job)
batch_size=1

# --- Prep ---
mkdir -p logs
declare -a combinations

# --- Build combinations ---
for seed in "${seeds[@]}"; do
  for env in "${envs[@]}"; do
    for noise in "${noise_levels[@]}"; do
      for n_feedback in "${n_feedbacks[@]}"; do
        for rr_loss_weight in "${rr_loss_weights[@]}"; do
          if [[ "${rr_loss_weight}" == "1.0" ]]; then
            combinations+=("$seed|$env|$noise|$n_feedback|$rr_loss_weight")
          else
            # No extra runs for non-RT; pass a neutral partitioner/size
            # (adjust 'none' and '0' if your script expects different sentinels)
            combinations+=("$seed|$env|$noise|$n_feedback|$rr_loss_weight")
          fi
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
#SBATCH --job-name=train_RL_agent_rr_${batch_id}
#SBATCH --time=02:00:00
#SBATCH --output=logs/train_RL_agent_${batch_id}_%j.out

EOT

  for item in "${batch[@]}"; do
    IFS='|' read -r seed env noise n_feedback rr_loss_weight <<< "$item"
    echo "python multi_type_feedback/train_RL_agent.py \\
      --algorithm ppo \\
      --environment \"$env\" \\
      --feedback-type \"comparative\" \\
      --n-feedback \"$n_feedback\" \\
      --seed \"$seed\" \\
      --noise-level \"$noise\" \\
      --rr-loss-weight \"$rr_loss_weight\" \\
      --wandb-project-name response_rank &" >> "$sbatch_script"
  done

  echo "wait" >> "$sbatch_script"
  sbatch "$sbatch_script"
  rm "$sbatch_script"
done

echo "All jobs have been submitted."
