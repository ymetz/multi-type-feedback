python -m venv venv
source venv/bin/activate
pip3 install torch torchvision
cd main && pip install -e .
cd procgen && pip install -e .
cd ../..
pip install -e stable-baselines3
cd rlhf
pip install -e masksembles
pip install imitation
pip install gymnasium==1.0.0a2 lightning minigrid mujoco ale-py wandb
