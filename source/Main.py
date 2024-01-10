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
            model = DQN("MultiInputPolicy", env, verbose=1, learning_rate=0.004, batch_size=256,
                        gamma=0.25, exploration_initial_eps=1.0, exploration_final_eps=0.01, exploration_fraction=0.99)
        model.learn(total_timesteps=2000000, log_interval=20, progress_bar=True)
        model.save("models/CustomGame_level1v3")

        del model  # remove to demonstrate saving and loading
    else:
        print("show")
        model = DQN.load("models/CustomGame_level1v3")

        obs, info = env.reset()

        i = 0
        max_i = Consts.NUM_EPISODES_TEST

        times = []
        rewards = []

        while i < max_i:
            action, _states = model.predict(obs, deterministic=False)
            obs, reward, terminated, truncated, info = env.step(action)
            env.render()
            if terminated or truncated:
                times.append(obs['time'])
                rewards.append(reward)

                obs, info = env.reset()
                i += 1

        sum_times = sum(times)
        sum_reward = sum(rewards)

        mean_time = sum_times / len(times)
        mean_reward = sum_reward / len(rewards)

        print(times, mean_time, rewards, mean_reward)

if __name__ == "__main__":
    main()
