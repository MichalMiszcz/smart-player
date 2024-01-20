import gymnasium as gym

from stable_baselines3 import DQN

import csv

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
            model = DQN.load(path=Consts.FILE_TO_LOAD, env=env, verbose=1)
        else:
            print("create model")
            model = DQN("MultiInputPolicy", env, verbose=1, learning_rate=0.1,
                        batch_size=256, gamma=0.25, exploration_initial_eps=1.0, exploration_final_eps=0.01,
                        exploration_fraction=0.5)
        time_steps = 50000
        model.learn(total_timesteps=time_steps, log_interval=20, progress_bar=True)
        model.save(path=Consts.FILE_TO_SAVE)

        del model  # remove to demonstrate saving and loading
    else:
        print("show")
        model = DQN.load(path=Consts.FILE_TO_LOAD)

        obs, info = env.reset()

        i = 0
        max_i = Consts.NUM_EPISODES_TEST

        times = []
        rewards = []
        messages = []
        reward_sum = 0

        while i < max_i:
            action, _states = model.predict(obs, deterministic=False)
            obs, reward, terminated, truncated, info = env.step(action)
            env.render()

            reward_sum += reward

            if terminated or truncated:
                times.append(obs['time'])
                rewards.append(reward_sum)
                messages.append(info['message'])
                reward_sum = 0

                obs, info = env.reset()
                i += 1

        sum_times = sum(times)
        sum_reward = sum(rewards)

        mean_time = sum_times / len(times)
        mean_reward = sum_reward / len(rewards)

        times_values = [str(arr[0]).replace('.', ',') for arr in times]
        reward_values = [str(arr).replace('.', ',') for arr in rewards]

        print(times_values, mean_time, reward_values, mean_reward)

        # Specify the CSV file path
        times_file_path = 'output/times.csv'
        rewards_file_path = 'output/rewards.csv'
        messages_file_path = 'output/messages.csv'

        # Writing to the CSV file
        with open(times_file_path, 'w', newline='') as csvfile:
            # Creating a CSV writer
            csv_writer = csv.writer(csvfile)

            # Writing the data
            csv_writer.writerows(zip(times_values))

        with open(rewards_file_path, 'w', newline='') as csvfile:
            # Creating a CSV writer
            csv_writer = csv.writer(csvfile)

            # Writing the data
            csv_writer.writerows(zip(reward_values))

        with open(messages_file_path, 'w', newline='') as csvfile:
            # Creating a CSV writer
            csv_writer = csv.writer(csvfile)

            # Writing the data
            csv_writer.writerows(zip(messages))


if __name__ == "__main__":
    main()
