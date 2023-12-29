import gymnasium as gym
from stable_baselines3 import DQN

import Consts

import CustomGameEnvironment

gym.envs.register(
    id='CustomGame-v0',
    entry_point='CustomGameEnvironment:CustomGameEnvironment',
)

env = gym.make('CustomGame-v0')


def main():

    if Consts.LEARN:
        if Consts.LOAD:
            print("load")
            model = DQN.load("models/CustomGame_v3", env, verbose=1)
        else:
            print("create model")
            model = DQN("MultiInputPolicy", env, verbose=1, learning_rate=0.01, batch_size=256,
                        gamma=0.99, exploration_initial_eps=1.0, exploration_final_eps=0.05, exploration_fraction=0.95)
        model.learn(total_timesteps=1000000, log_interval=20, progress_bar=True)
        model.save("models/CustomGame_v2")

        del model  # remove to demonstrate saving and loading
    else:
        print("show")
        model = DQN.load("models/CustomGame_v2")

        obs, info = env.reset()
        while True:
            action, _states = model.predict(obs, deterministic=False)
            obs, reward, terminated, truncated, info = env.step(action)
            env.render()
            if terminated or truncated:
                obs, info = env.reset()


if __name__ == "__main__":
    main()
