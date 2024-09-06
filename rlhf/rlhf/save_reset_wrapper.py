import gymnasium as gym
from gymnasium.envs.mujoco import MujocoEnv
from rl_zoo3.wrappers import Gym3ToGymnasium
from ale_py import AtariEnv
from minigrid.minigrid_env import MiniGridEnv
import copy
import numpy as np

class SaveResetEnvWrapper(gym.Wrapper):
    def __init__(self, env):
        super(SaveResetEnvWrapper, self).__init__(env)

    def save_state(self, observation=None):
        """
        Save the current state of the environment and return it.
        For MuJoCo environments, use the env.sim.get_state() method.
        For MiniGrid environments, use copy.deepcopy() to save the state.
        """
        if isinstance(self.unwrapped, MujocoEnv):
            # MuJoCo environment
            state = {"qpos": np.copy(self.unwrapped.data.qpos), "qval": np.copy(self.unwrapped.data.qvel)}
        elif isinstance(self.unwrapped, AtariEnv):
            state = self.unwrapped.clone_state()
        elif isinstance(self.unwrapped, MiniGridEnv):
            # Minigrid environment
            state = {
                "grid": self.unwrapped.grid,
                "carrying": self.unwrapped.carrying,
                "agent_pos": self.unwrapped.agent_pos,
                "agent_dir": self.unwrapped.agent_dir,
                "mission": self.unwrapped.mission
            }
        elif isinstance(self.unwrapped, Gym3ToGymnasium):
            state = self.unwrapped.env.get_state()
        else:
            # Something else
            state = copy.deepcopy(self.unwrapped)

        return {"state": state, "observation": observation}

    def load_state(self, state_and_obs):
        """
        Load a given state into the environment.
        For MuJoCo environments, use the env.sim.set_state() method.
        For MiniGrid environments, directly assign the provided state.
        Returns the observation
        """
        if state_and_obs is None:
            raise ValueError("The provided state is None. Please provide a valid state.")

        obs = state_and_obs["observation"]
        state = state_and_obs["state"]
        if isinstance(self.unwrapped, MujocoEnv):
            # MuJoCo environment
            self.unwrapped.set_state(state["qpos"], state["qval"])
        elif isinstance(self.unwrapped, AtariEnv):
            self.unwrapped.restore_state(state)
        elif isinstance(self.unwrapped, MiniGridEnv):
            # Minigrid environment (A bit cluncky i guess)
            self.unwrapped.grid = state["grid"]
            self.unwrapped.carrying = state["carrying"]
            self.unwrapped.agent_pos = state["agent_pos"]
            self.unwrapped.agent_dir = state["agent_dir"]
            self.unwrapped.mission = state["mission"]
        elif isinstance(self.unwrapped, Gym3ToGymnasium):
            self.unwrapped.env.set_state(state)
        else:
            # Something else
            self.unwrapped = state
        return obs

    def reset(self, **kwargs):
        """
        Reset the environment. If a state is provided, load it after resetting.
        Otherwise, perform a normal reset.
        """
        observation, info = super(SaveResetEnvWrapper, self).reset(**kwargs)
        return observation, info
