# Multi-Type-Feedback

This repository contains code for training and evaluating reinforcement learning agents using various types of feedback.

## Repository Structure

- `main/`: Main training scripts (Main is a fork of `Stable Baselines3 Zoo`, not by the authors of this repository)
- `stable-baselines3/`: A slightly modified version of Stable Baselines (fix with gymnasium==1.0.0a2), not by the authors of this repository
- `rlhf/`: Scripts for reward model training and agent training with learned rewards
- `setup.sh`: Setup script for the environment
- `rlhf/masksembles/`: Masksembles implementation, not by the authors of this repository

## Main Components

### 1. Initial Training (`main/train.py`)

Trains PPO agents on various environments:

```bash
python train.py --algo ppo --env <environment> --verbose 0 --save-freq <frequency> --seed <seed> --gym-packages procgen ale_py --log-folder gt_agents
```

Environments: Ant-v5, Swimmer-v5, HalfCheetah-v5, Hopper-v5, Atari, Procgen, ...

### 2. Feedback Generation (`rlhf/generate_feedback.py`)

Generates feedback for trained agents:

```bash
python rlhf/generate_feedback.py --algorithm ppo --environment <env> --seed <seed> --n-feedback 10000 --save-folder feedback_regen
```

Note: The script looks in the gt_agents folder for trained agents. Abd expects that the `python main/benchmark_envs.py` script has been run to generate the evaluation scores.

### 3. Reward Model Training (`rlhf/train_reward_model.py`)

Trains reward models based on generated feedback:

```bash
python rlhf/train_reward_model.py --algorithm ppo --environment <env> --feedback-type <type> --seed <seed>
```

Feedback types: evaluative, comparative, demonstrative, corrective, descriptive, descriptive_preference

### 4. Agent Training with Learned Rewards (`rlhf/train_agent.py`)

Trains agents using the learned reward models:

```bash
python rlhf/train_agent.py --algorithm ppo --environment <env> --feedback-type <type> --seed <seed>
```

### 5. Agent Training with Learned Reward Function Ensemble (`rlhf/rlhf/train_agent_ensemble.py`)

Trains agents using the learned reward models:

```bash
python rlhf/train_agent_ensemble.py --algorithm ppo --environment <env> --feedback-types <types> --seed <seed>
```

Feedback types: evaluative, comparative, demonstrative, corrective, descriptive, descriptive_preference

## Usage

1. Set up the environment using `setup.sh`
2. Run initial training (e.g. with `main/start_training.sh`)
3. Generate feedback
4. Train reward models
5. Train agents with learned rewards

For detailed parameters and options, refer to the individual script files.


## Additional files for figure generation and plotting

- `main/benchmark_envs.py`: Benchmark trained agents on various environments
- `rlhf/Analyze_Generated_Feedback.ipynb`: Jupyter notebook for analyzing generated feedback
- `rlhf/Analyze_Reward_Model_Predictions.ipynb`: Jupyter notebook for analyzing reward models
- `rlhf/Generate_RL_result_curves.ipynb`: Jupyter notebook for generating RL result curves

and more...


## Supported Environments

- Mujoco
- Procgen
- Atari
- potentially other Gym environments

## Notes

- This repository uses CUDA for GPU acceleration. Ensure proper CUDA setup before running.
- The training scripts are designed to distribute jobs across multiple GPUs.
- For large-scale experiments, consider using a job scheduler like Slurm (example scripts provided in the original bash files).