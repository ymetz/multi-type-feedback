# ResponseRank: Data-Efficient Reward Modeling through Preference Strength Learning

**Paper:** [ResponseRank: Data-Efficient Reward Modeling through Preference Strength Learning](https://arxiv.org/abs/2512.25023)

This repository contains the code for the control experiments presented in the ResponseRank paper. While the codebase builds on existing training infrastructure for multi-type feedback, **ResponseRank** focuses exclusively on comparative (pairwise preference) feedback and employs a novel RtRank loss that incorporates response information to improve reward modeling efficiency.

## Table of Contents

- [Repository Structure](#repository-structure)
- [Main Components](#main-components)
  - [Initial Training](#1-initial-training-train_baselinestrainpy)
  - [Feedback Generation](#2-feedback-generation-multi_type_feedbackgenerate_feedbackpy)
  - [Reward Model Training](#3-reward-model-training-multi_type_feedbacktrain_reward_modelpy)
  - [Agent Training with Learned Rewards](#4-agent-training-with-learned-rewards-multi_type_feedbacktrain_rl_agentpy)
- [Quick Start](#quick-start)
- [Analysis and Visualization](#analysis-and-visualization)
- [Replication](#replication)
- [Supported Environments](#supported-environments)
- [Technical Notes](#technical-notes)
- [Citation](#citation)

## Repository Structure

- `train_baselines/`: Training scripts for expert models (Main is a fork of `Stable Baselines3 Zoo`, not by the authors of this repository) - used for sampling of datasets
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

**Environments used in the paper:** HalfCheetah-v5, Swimmer-v5, Walker2d-v5 (MuJoCo), and highway-fast-v0, merge-v0 (Highway-env), all trained with PPO.

For future experiments, the framework already supports SAC. We recommend the training of more complex environments including Ant-v5, Hopper-v5, Humanoid-v5 and Metaworld environments with SAC:

```bash
python train_baselines/train.py --algo sac --env <Ant-v5|Hopper-v5|metaworld-sweep-into-v2|...> --verbose 0 --save-freq <frequency> --seed <seed> --gym-packages procgen ale_py --log-folder gt_agents
```

**Note:** Use `gt_agents` as the log folder to ensure compatibility with downstream scripts. You can adapt the expert model directories in the generation script if needed.

### 2. Feedback Generation (`multi_type_feedback/generate_feedback.py`)

Generates feedback for trained agents:

```bash
python multi_type_feedback/generate_feedback.py --algorithm ppo --environment <env> --seed <seed> --n-feedback 5000 --save-folder feedback
```

**Note:** This script loads trained agents from the `gt_agents` folder and expects that `python train_baselines/train_baselines/benchmark_evals.py` has been run to generate evaluation scores for the agents.

### 3. Reward Model Training (`multi_type_feedback/train_reward_model.py`)

Trains reward models based on generated feedback using either standard Bradley-Terry (BT) loss or the **ResponseRank (RtRank)** loss.

#### Basic Usage:

```bash
python multi_type_feedback/train_reward_model.py --algorithm ppo --environment <env> --seed <seed> --feedback-folder feedback --save-folder reward_models
```

By default, reward models are trained with comparative (pairwise preference) feedback using the standard Bradley-Terry loss.

#### ResponseRank Training:

To enable **ResponseRank** training with response-aware loss, specify the `--rt-loss-weight` parameter:

```bash
python multi_type_feedback/train_reward_model.py --algorithm ppo --environment <env> --seed <seed> --feedback-folder feedback --save-folder reward_models --rt-loss-weight 0.5
```

#### Key ResponseRank Parameters:

- `--rt-loss-weight <float>`: Weight for the RT-rank loss component (default: 0.0 for standard BT loss).
  - `0.0`: Standard Bradley-Terry loss (baseline)
  - `0.1-1.0`: ResponseRank loss with increasing emphasis on response information

- `--stratifier <str>`: Stratification method for grouping preferences by similarity (default: `global`). Options:
  - `global`: Single global partition across all preferences
  - `knn`: K-nearest neighbors clustering based on trajectory features
  - `std_window`: Stratification based on standard deviation windows
  - `none`: No stratification

- `--partitioner <str>`: Partitioning strategy for batch construction (default: `random`). Options:
  - `random`: Random assignment to partitions of specified size
  - `round_robin`: Distribute examples evenly across partitions
  - `none`: No partitioning

- `--partition-size <int>`: Target size for each partition when using partitioner (default: 8)

- `--n-ensemble <int>`: Number of ensemble models for uncertainty estimation (default: 1)

#### Example with Full ResponseRank Configuration:

```bash
python multi_type_feedback/train_reward_model.py \
    --algorithm ppo \
    --environment HalfCheetah-v5 \
    --seed 42 \
    --feedback-folder feedback \
    --save-folder reward_models \
    --rt-loss-weight 1.0  \
    --stratifier global \
    --partitioner random \
    --partition-size 8 \
    --n-ensemble 3
```


### 4. Agent Training with Learned Rewards (`multi_type_feedback/train_RL_agent.py`)

Trains RL agents using the learned reward models instead of the ground-truth environment rewards:

```bash
python multi_type_feedback/train_RL_agent.py --algorithm ppo --environment <env> --seed <seed>
```

This script automatically loads the corresponding trained reward model from the `reward_models` folder and uses it to provide rewards during training. You can load reward models trained via the BT baseline or via RT weight by passing the `rt-loss-weight`parameter analogously to reward model training. If available, the corresponding reward model is automatically loaded:

```bash
python multi_type_feedback/train_RL_agent.py --algorithm ppo --environment <env> --seed <seed> --rt-loss-weight 1.0
```

## Quick Start

### Installation

```bash
pip install -e .
```

Ensure CUDA is properly configured for GPU acceleration.

### Basic Workflow

1. **Train Expert Agents**: Train PPO/SAC agents on your target environment
   ```bash
   python train_baselines/train.py --algo ppo --env HalfCheetah-v5 --seed 0 --log-folder gt_agents
   ```

2. **Generate Preference Feedback**: Create pairwise comparison dataset
   ```bash
   python multi_type_feedback/generate_feedback.py --algorithm ppo --environment HalfCheetah-v5 --seed 0 --n-feedback 5000 --save-folder feedback
   ```

3. **Train Reward Model**: Train using standard BT or ResponseRank loss
   ```bash
   # Standard BT loss
   python multi_type_feedback/train_reward_model.py --algorithm ppo --environment HalfCheetah-v5 --seed 0

   # ResponseRank loss
   python multi_type_feedback/train_reward_model.py --algorithm ppo --environment HalfCheetah-v5 --seed 0 --rt-loss-weight 0.5 --stratifier global
   ```

4. **Train RL Agent with Learned Rewards**: Use the learned reward model
   ```bash
   python multi_type_feedback/train_RL_agent.py --algorithm ppo --environment HalfCheetah-v5 --seed 0
   ```

For detailed parameters and options, refer to the individual script files or use the `--help` flag.


## Analysis and Visualization

The repository includes several Jupyter notebooks for analyzing results and generating figures:

- [notebooks/RtRank_Generate_Data_and_Plots.ipynb](notebooks/RtRank_Generate_Data_and_Plots.ipynb) Read data from W&B to generate result tables and plots (learning curves)
- [train_baselines/train_baselines/benchmark_evals.py](train_baselines/train_baselines/benchmark_evals.py): Benchmark trained agents on various environments
- [notebooks/Analyze_Generated_Feedback.ipynb](notebooks/Analyze_Generated_Feedback.ipynb): Analyze generated preference feedback datasets
- [notebooks/Analyze_Reward_Model_Predictions.ipynb](notebooks/Analyze_Reward_Model_Predictions.ipynb): Evaluate reward model predictions and accuracy
- [notebooks/Generate_RL_result_curves.ipynb](notebooks/Generate_RL_result_curves.ipynb): Generate learning curves for RL agents

Additional analysis notebooks are available in the `notebooks/` directory.

# Replication

To replicate the results reported in the paper, we have prepared four independent scripts. We utilize Weights & Biases for data logging and
synchronization. To generate the result files (tables & plots), run the following scripts:

1. Train expert models
```bash
./scripts/train_expert_models_rt.sh
```

2. Generate feedback datasets based on expert model-sampling
```bash
./scripts/generate_feedback_rt.sh
```

3. Train reward models based on generated feedback
```bash
./scripts/train_reward_models_rt.sh
```

4. Train downstream RL models based on reward models
```bash
./scripts/train_RL_models_rt.sh
```

To generate equivalent results, be aware to use the provided default values for stratification, and partitioning:

```python
    parser.add_argument(
        "--stratifier",
        type=str,
        default="global",
        choices=["knn", "std_window", "global", "none"],
        help="Stratification method for RT-rank loss",
    )
    parser.add_argument(
        "--partitioner",
        type=str,
        default="random",
        choices=["round_robin", "random", "none"],
    )
    parser.add_argument(
        "--partition-size",
        type=int,
        default=8,
    )
```

The provided scripts are targeted for a `SLURM`-based job submission system. Use your AI tool of choice to translate it
to alternative job scheduling systems.


## Supported Environments

- **MuJoCo**: HalfCheetah-v5, Swimmer-v5, Walker2d-v5, Ant-v5, Hopper-v5, Humanoid-v5
- **Highway-env**: highway-fast-v0, merge-v0, roundabout-v0
- **Metaworld**: metaworld-sweep-into-v2, metaworld-button-press-v2, metaworld-pick-place-v2
- **Procgen**: Available in `dependencies/procgen` (fork with Gymnasium support)
- **Atari**: ALE environments (BeamRider, MsPacman, Enduro, Pong, etc.)
- Other Gym/Gymnasium-compatible environments

## Technical Notes

- **GPU Acceleration**: This repository requires CUDA for GPU acceleration. Ensure proper CUDA setup before running experiments.
- **Logging**: Experiments are tracked using Weights & Biases (wandb). Configure your wandb project name using the `--wandb-project-name` flag.
- **Reproducibility**: Set seeds explicitly using the `--seed` parameter for reproducible experiments.
- **Job Scheduling**: For large-scale experiments, consider using a job scheduler like Slurm. Example scripts are available in the bash files.

## Citation

If you use this code in your research, please cite the ResponseRank paper:

```bibtex
@inproceedings{kaufmann2025responserank,
  slug = {responserank},
  details_link = {true},
  custom_content_before = {true},
  author = {Kaufmann, Timo and Metz, Yannick and Keim, Daniel and Hüllermeier, Eyke},
  booktitle = {The Annual Conference on Neural Information Processing Systems (NeurIPS)},
  title = {ResponseRank: Data-Efficient Reward Modeling through Preference Strength Learning},
  year = {2025}
}
```

## License

[Add license information here]

## Acknowledgments

This repository builds upon:
- [Stable Baselines3 Zoo](https://github.com/DLR-RM/rl-baselines3-zoo) for baseline training
- [Imitation](https://github.com/HumanCompatibleAI/imitation) package (adapted version included in `dependencies/`)
- Masksembles implementation (included in `dependencies/masksembles/`)
