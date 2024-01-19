import os

import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt

from stable_baselines3 import DQN
from stable_baselines3.common import results_plotter
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.results_plotter import load_results, ts2xy, plot_results
from stable_baselines3.common.callbacks import BaseCallback

import csv
import matplotlib.pyplot as plt

import Consts
import CustomGameEnvironment


# Create log dir
log_dir = "logs/"
os.makedirs(log_dir, exist_ok=True)

gym.envs.register(
    id='CustomGame-v0',
    entry_point='CustomGameEnvironment:CustomGameEnvironment',
)

env = gym.make('CustomGame-v0')
env = Monitor(env, log_dir)


class SaveOnBestTrainingRewardCallback(BaseCallback):
    """
    Callback for saving a model (the check is done every ``check_freq`` steps)
    based on the training reward (in practice, we recommend using ``EvalCallback``).

    :param check_freq:
    :param log_dir: Path to the folder where the model will be saved.
      It must contains the file created by the ``Monitor`` wrapper.
    :param verbose: Verbosity level: 0 for no output, 1 for info messages, 2 for debug messages
    """

    def __init__(self, check_freq: int, log_dir: str, verbose: int = 1):
        super().__init__(verbose)
        self.check_freq = check_freq
        self.log_dir = log_dir
        self.save_path = os.path.join(log_dir, "best_model")
        self.best_mean_reward = -np.inf

    def _init_callback(self) -> None:
        # Create folder if needed
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq == 0:

            # Retrieve training reward
            x, y = ts2xy(load_results(self.log_dir), "timesteps")
            if len(x) > 0:
                # Mean training reward over the last 100 episodes
                mean_reward = np.mean(y[-100:])
                if self.verbose >= 1:
                    print(f"Num timesteps: {self.num_timesteps}")
                    print(
                        f"Best mean reward: {self.best_mean_reward:.2f} - Last mean reward per episode: {mean_reward:.2f}")

                # New best model, you could save the agent here
                if mean_reward > self.best_mean_reward:
                    self.best_mean_reward = mean_reward
                    # Example for saving best model
                    if self.verbose >= 1:
                        print(f"Saving new best model to {self.save_path}")
                    self.model.save(self.save_path)

        return True


class RewardLoggerCallback(BaseCallback):
    def __init__(self, verbose=0):
        super(RewardLoggerCallback, self).__init__(verbose)
        self.cumulative_rewards = []

    def _on_step(self) -> bool:
        print(self.locals['rewards'][0])

        if 'rewards' in self.locals:
            self.cumulative_rewards.append(self.locals['rewards'][0])
        return True


def main():
    if Consts.LEARN:
        callback = SaveOnBestTrainingRewardCallback(check_freq=500, log_dir=log_dir)
        # reward_logger = RewardLoggerCallback()

        if Consts.LOAD:
            print("load")
            model = DQN.load(path=Consts.FILE_TO_LOAD, env=env, verbose=1)
        else:
            print("create model")
            model = DQN("MultiInputPolicy", env, verbose=1, learning_rate=0.1,
                        batch_size=256, gamma=0.25, exploration_initial_eps=1.0, exploration_final_eps=0.01,
                        exploration_fraction=0.5)
        time_steps = 50000
        model.learn(total_timesteps=time_steps, log_interval=20, progress_bar=True, callback=callback)
        model.save(path=Consts.FILE_TO_SAVE)

        plot_results([log_dir], time_steps, results_plotter.X_TIMESTEPS, "ML Agent")
        plt.show()

        # plt.plot(reward_logger.cumulative_rewards)
        # plt.xlabel('Steps')
        # plt.ylabel('Cumulative Reward')
        # plt.title('Cumulative Reward Over Time')
        # plt.show()

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
