from setuptools import setup, find_packages

setup(
    name="multi-type-feedback",
    version="0.1.0",
    description="Reward Learning from Multiple Feedback Types",
    author="András Geiszl",
    author_email="geiszla@gmail.com",
    license="UNLICENSED",
    packages=find_packages(),
    install_requires=[
        "torch",
        "torchvision",
        "imitation",
        "gymnasium",
        "lightning",
        "minigrid",
        "mujoco",
        "ale-py",
        "wandb",
    ],
    extras_require={
        "procgen": ["procgen @ file://dependencies/procgen"],
    },
    entry_points={
        "console_scripts": [
            "generate_feedback=rlhf.generate_feedback:main",
            "train_reward=rlhf.train_reward_model:main",
            "train_agent=rlhf.train_agent:main",
            "plot_reward=rlhf.plot_reward_model_output:main",
            "export_videos=rlhf.export_videos:main",
        ],
    },
    python_requires=">=3.11",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: Other/Proprietary License",
        "Programming Language :: Python :: 3.11",
    ],
)