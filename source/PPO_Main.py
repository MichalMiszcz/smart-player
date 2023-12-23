import gymnasium as gym

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

import CustomGameEnvironment

gym.envs.register(
    id='CustomGame-v0',
    entry_point='CustomGameEnvironment:CustomGameEnvironment',
)


def main():
    # Parallel environments
    vec_env = gym.make('CustomGame-v0')

    model = PPO("MultiInputPolicy", vec_env, verbose=1)
    model.learn(total_timesteps=25000)
    model.save("models/ppo_custom")

    del model  # remove to demonstrate saving and loading

    model = PPO.load("models/ppo_custom")

    obs = vec_env.reset()
    while True:
        print("obs shape:", obs)
        action, _states = model.predict(obs)
        obs, rewards, dones, info = vec_env.step(action)
        vec_env.render("human")


if __name__ == "__main__":
    main()
