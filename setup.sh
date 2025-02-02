python -m venv venv
source venv/bin/activate
pip install torch torchvision
cd main && pip install -e .
pip install -e procgen
cd ..
cd rlhf
pip install -e masksembles
cd ..
pip install imitation
pip install -e stable-baselines3
pip install gymnasium lightning minigrid mujoco ale-py wandb
