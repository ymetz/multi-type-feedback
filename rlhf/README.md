# Machine Learning Project

This is the repository for the Masters thesis project on Reinforcement Learning from Human Feedback.

## Setting up

*Note: if you are using Windows, you will need to either use a Linux VM or WSL (see [WSL setup instructions](#setting-up-wsl-recommended-for-windows) below).*

### Setting up WSL (recommended for Windows)

1. Set up [WSL](https://learn.microsoft.com/en-us/windows/wsl/install)
1. Open WSL command line and clone the repo using it (to a path NOT starting with `/mnt/`) instead of using a Windows command prompt or PowerShell. This will make the development environment faster.
1. Add `export DISPLAY=$(cat /etc/resolv.conf | grep nameserver | awk '{print $2}'):0` to your shell profile (e.g., to the end of `~/.bashrc` or `~/.zshrc` or similar) to enable WSL to open windows and display GUIs.
1. If you are using VSCode, run `code .` inside the project directory to open it (or if you've opened the project before, you can access it from `File -> Open Recent`). See [Open a WSL project in Visual Studio Code](https://learn.microsoft.com/en-us/windows/wsl/tutorials/wsl-vscode#open-a-wsl-project-in-visual-studio-code) for more details.
1. Do all further setup inside the WSL command line or from the terminal of VSCode opened from WSL.

### Installing dependencies

1. Install Python 3.11
1. Run `poetry install` in the project directory to install dependencies of the project. It will create a separate environment for this project and activate it every time you run a command with `poetry run`.
1. To install MuJoCo, follow the [instructions in the GitHub repo](https://github.com/openai/mujoco-py/#install-mujoco).
1. Add `export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:~/.mujoco/mujoco210/bin` to your shell profile and start a new shell to make MuJoCo discoverable.
1. Make sure all the required libraries are installed by running `sudo apt install gcc libosmesa6-dev libgl1 libglfw3 patchelf`
1. If you don't have `ffmpeg` installed yet, [install it on your system](https://ffmpeg.org/download.html) or run `pip install imageio-ffmpeg` to install it in the project locally.

### Setting up VSCode (recommended)

1. Install and open [VSCode](https://code.visualstudio.com/download)
1. Install these VSCode extensions (by searching for them on the extensions tab): `charliermarsh.ruff`, `njpwerner.autodocstring`, `visualstudioexptteam.vscodeintellicode`, `ms-python.black-formatter`, `ms-python.isort`, `ms-python.vscode-pylance`, `ms-python.pylint`, `ms-python.python`, `kevinrose.vsc-python-indent`, `tamasfe.even-better-toml`
1. Open the command palette, choose `Python: Select Interpreter`, then select the virtual environment created by Poetry.

   *Note: If the desired environment is not in the list, you can find the location of the environments by running `poetry env info -p`, then add the interpreter as a new entry.*
1. Start a new terminal. VSCode will automatically activate the selected environment.

## Downloading the expert model

To download the expert model, run `poetry run python -m rl_zoo3.load_from_hub --algo sac --env HalfCheetah-v3 -orga sb3 -f experts/` after activating the mamba environment.

## Running the code

You can run scripts specified in `pyproject.toml` with `poetry run <script name>`. For example, to run the `train_reward` script, run `poetry run train_reward` (you might also need to run `poetry install` before to update the dependencies).

### Reproducing the results

*Note: Some of the results are included in `reward_model_checkpoints` (trained reward models), `rl_checkpoints` (trained RL agents) and `rl_logs` (TensorBoard RL training logs).*

1. Open `rlhf/common.py` and set/increment the `EXPERIMENT_NUMBER` and the `FEEDBACK_TYPE` for the experiment you want to run. The experiment number will be appended to the beginning of logs and output files.
   *Note: For changing the expert, some parts of scripts currently commented out are need to be added back in.*
1. Log into Weights and Biases by running `poetry run wandb login`.
1. Make sure that the expert model is [downloaded](#downloading-the-expert-model), and run `poetry run generate_feedback` to generate data to train the feedback models. If successful, this will create a `.pkl` file inside the `feedback` directory in the project root directory.
1. Run `poetry run train_reward` to train the reward model for the selected feedback. If successful, this will save the best reward model checkpoint in the `reward_model_checkpoints` directory suffixed by a random number. You will need to use this suffix to refer to this model in the next steps.
1. Run `poetry run train_agent [model suffixes]`, where `[model suffixes]` are the feedback types and corresponding random numbers generated in the previous step (e.g., `evaluative-1869 descriptive-5890`), to train the RL agent using the selected reward model. If multiple models are specified, they will be combined to predict the reward.
1. Run `poetry run tensorboard --logdir rl_logs` to follow the training progress of the agent in TensorBoard.

#### Useful scripts

- `poetry run plot_reward [model suffix]` - plots the reward model's predictions against the true reward for the generated feedback data. You can edit the checkpoint used for the plot and the number of steps plotted in the script.
- `poetry run export_videos [model suffix]` - exports videos of the trained agent's performance in the environment. You can edit the length and number of videos exported in the script.
