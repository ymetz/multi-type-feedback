# Reward Learning from Multiple Feedback Types

Code for Feedback Generation, Reward Learning, and RL Benchmarks

This repository contains code for training and evaluating reinforcement learning agents using various types of feedback.

## Repository Structure

- `train_baselines/`: Main training scripts (Main is a fork of `Stable Baselines3 Zoo`, not by the authors of this repository)
- `multi_type_feedback/`: Scripts for reward model training and agent training with learned rewards
- `setup.sh`: Setup script for the environment
- `dependencies/masksembles/`: Masksembles implementation, not by the authors of this repository
- `dependencies/imitation/`: Slightly adapted/stripped down version of the [imitation package](https://github.com/HumanCompatibleAI/imitation)

## Main Components

### 1. Initial training (`train_baselines/train.py`)

Trains PPO agents in various environments:

```bash
python train_baselines/train.py --algo ppo --env <environment> --verbose 0 --save-freq <frequency> --seed <seed> --gym-packages procgen ale_py --log-folder gt_agents
```

In the paper, we trained the Mujoco environments HalfCheetah-v5, Swimmer-v5, and Walker2d-v5, as well as the Highway-env environments with PPO.
We recommend to train other continuous control environments, including Ant-v5, Hopper-v5, Humanoid-v5 and Metaworld environments with SAC:

```bash
python train_baselines/train.py --algo sac --env <Ant-v5|Hopper-v5|metaworld-sweep-into-v2|...> --verbose 0 --save-freq <frequency> --seed <seed> --gym-packages procgen ale_py --log-folder gt_agents
```

Environments: Ant-v5, Swimmer-v5, HalfCheetah-v5, Hopper-v5, Atari, Procgen, ...
Info: Please use gt_agents as the log folder to ensure compatibility with the generation script. However, you can adapt the expert model dirs if necessary.

### 2. Feedback Generation (`multi_type_feedback/generate_feedback.py`)

Generates feedback for trained agents:

```bash
python multi_type_feedback/generate_feedback.py --algorithm ppo --environment <env> --seed <seed> --n-feedback 10000 --save-folder feedback
```

Note: The script looks in the gt_agents folder for trained agents. It expects that the `python train_baselines/train_baselines/benchmark.py` script has been run to generate the evaluation scores.

### 2.5 Random Samples for Demonstrative Feedback (`multi_type_feedback/generate_samples.py`)

For demonstrative feedback, we utilize random feedback for simulated preference pairs. To generate the random samples, you need to run a separate script:

```bash
python multi_type_feedback/generate_samples.py --algorithm <ppo|sac> --environment HalfCheetah-v5 --n-samples 5000 --random
```

This script generates random samples in a separate `samples` directory. If you encounter errors like `FileNotFoundError: [Errno 2] No such file or directory: 'samples/random_HalfCheetah-v5.pkl`, be sure first to run the random sample generation script.

### 3. Reward Model Training (`multi_type_feedback/train_reward_model.py`)

Trains reward models based on generated feedback:

```bash
python multi_type_feedback/train_reward_model.py --algorithm ppo --environment <env> --feedback-type <type> --seed <seed> --feedback-folder feedback --save-folder reward_models
```

Feedback types: evaluative, comparative, demonstrative, corrective, descriptive, descriptive_preference

### 4. Agent Training with Learned Rewards (`multi_type_feedback/train_agent.py`)

Trains agents using the learned reward models:

```bash
python multi_type_feedback/train_RL_agent.py --algorithm ppo --environment <env> --feedback-type <type> --seed <seed>
```

### 5. Agent Training with Learned Reward Function Ensemble (`multi_type_feedback/train_agent_ensemble.py`)

Trains agents using the learned reward models:

```bash
python multi_type_feedback/train_RL_agent_with_ensemble.py --algorithm ppo --environment <env> --feedback-types <types> --seed <seed>
```

Feedback types: evaluative, comparative, demonstrative, corrective, descriptive, descriptive_preference

## Usage

1. Install the package using `pip install -e .`
2. Run initial training (e.g., with `train_baselines/start_training.sh`)
3. Generate feedback
4. Train reward models
5. Train agents with learned rewards

For detailed parameters and options, refer to the individual script files.


## Additional files for figure generation and plotting

- `train_baselines/benchmark.py`: Benchmark trained agents on various environments
- `multi_type_feedback/Analyze_Generated_Feedback.ipynb`: Jupyter notebook for analyzing generated feedback
- `multi_type_feedback/Analyze_Reward_Model_Predictions.ipynb`: Jupyter notebook for analyzing reward models
- `multi_type_feedback/Generate_RL_result_curves.ipynb`: Jupyter notebook for generating RL result curves

and more...


## Supported Environments

- Mujoco
- Procgen available in `dependency/procgen`: Fork of Procgen with Gymnasium support
- Atari
- potentially other Gym environments

## Notes

- This repository uses CUDA for GPU acceleration. Ensure proper CUDA setup before running.
- The training scripts are designed to distribute jobs across multiple GPUs.
- For large-scale experiments, consider using a job scheduler like Slurm (example scripts provided in the original bash files).
