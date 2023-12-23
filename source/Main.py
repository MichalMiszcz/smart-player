import gymnasium as gym
from stable_baselines3 import DQN

import CustomGameEnvironment

gym.envs.register(
    id='CustomGame-v0',
    entry_point='CustomGameEnvironment:CustomGameEnvironment',
)

env = gym.make('CustomGame-v0')


def main():
    model = DQN("MultiInputPolicy", env, verbose=1)
    model.learn(total_timesteps=10000, log_interval=4)
    model.save("models/CustomGame_v0")

    del model  # remove to demonstrate saving and loading

    model = DQN.load("models/CustomGame_v0")

    obs, info = env.reset()
    while True:
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            obs, info = env.reset()


if __name__ == "__main__":
    main()
