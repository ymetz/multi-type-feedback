import pickle
import gymnasium as gym
from pathlib import Path
from tqdm import tqdm
from metaworld.policies import ENV_POLICY_MAP

N_EPISODES = 200  # ~10k steps per task, assuming avg ep len ~50
OUT_DIR = Path("expert_data_mt50")
OUT_DIR.mkdir(exist_ok=True, parents=True)

def collect_expert_data(task_name: str, policy_cls, max_episodes: int = N_EPISODES):
    env = gym.make('Meta-World/MT1', env_name=task_name)
    policy = policy_cls()
    data = []

    for _ in tqdm(range(max_episodes), desc=f"Collecting for {task_name}"):
        obs, info = env.reset()
        episode = []

        for _ in range(500):  # Hard limit for episode length
            action = policy.get_action(obs)
            next_obs, _, _, _, info = env.step(action)

            episode.append((obs, action, next_obs, int(info['success'])))

            obs = next_obs
            if info['success']:
                break

        data.append(episode)

    with open(OUT_DIR / f"{task_name}.pkl", "wb") as f:
        pickle.dump(data, f)

if __name__ == "__main__":
    for task, policy_cls in ENV_POLICY_MAP.items():
        collect_expert_data(task, policy_cls)
