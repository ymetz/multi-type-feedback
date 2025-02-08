import os

from setuptools import find_packages, setup

# Get absolute paths to dependencies
base_dir = os.path.abspath(os.path.dirname(__file__))


def get_abs_path(rel_path):
    return f"file://{os.path.join(base_dir, rel_path)}"


setup(
    name="multi-type-feedback",
    version="0.1.0",
    description="Reward Learning from Multiple Feedback Types",
    author="YANNICK Metz",
    author_email="yannick.metz@uni-konstanz.de",
    license="MIT",
    packages=find_packages(),
    install_requires=[
        "torch",
        "torchvision",
        "gymnasium==1.0.0",
        "lightning",
        "minigrid",
        "mujoco",
        "ale-py",
        "wandb",
        "sb3-contrib",
        "metaworld @ git+https://github.com/Farama-Foundation/Metaworld.git@master#egg=metaworld",
        "gym3",
        "procgen @ git+https://github.com/juancroldan/procgen",
        f"stable-baselines3 @ {get_abs_path('dependencies/stable-baselines3')}",
        f"imitation @ {get_abs_path('dependencies/imitation')}",
        f"masksembles @ {get_abs_path('dependencies/masksembles')}",
    ],
    python_requires=">=3.11",
)
