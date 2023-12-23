import random
import time

import arcade
import arcade.gui
import gymnasium as gym
from gymnasium.vector.utils import spaces

import Consts
import CustomGameEnvironment

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

gym.envs.register(
    id='CustomGame-v0',
    entry_point='CustomGameEnvironment:CustomGameEnvironment',
)


# Define the Deep Q Network (DQN) model
class DQN(nn.Module):
    def __init__(self, input_size, output_size):
        super(DQN, self).__init__()
        self.fc = nn.Linear(input_size, 128)
        self.relu = nn.ReLU()
        self.output_layer = nn.Linear(128, output_size)

    def forward(self, x):
        x = self.fc(x)
        x = self.relu(x)
        return self.output_layer(x)


# Replay Buffer to store experiences for training
class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, transition):
        if len(self.memory) < self.capacity:
            self.memory.append(transition)
        else:
            self.memory[self.position] = transition
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return zip(*random.sample(self.memory, batch_size))


# DQN Agent
class DQNAgent:
    def __init__(self, input_size, output_size, learning_rate=0.01, gamma=0.9, epsilon=0.1):
        self.q_network = DQN(input_size, output_size)
        self.target_network = DQN(input_size, output_size)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)
        self.gamma = gamma
        self.epsilon = epsilon
        self.replay_buffer = ReplayBuffer(capacity=10000)

    def select_action(self, state):
        state_vector = self.flatten_state(state)
        state_tensor = torch.FloatTensor(state_vector)

        if np.random.rand() < self.epsilon:
            return np.random.choice(Consts.NUMBER_OF_ACTIONS)  # tymczasowe rozwiazanie, bo jest 5 akcji

        with torch.no_grad():
            q_values = self.q_network(state_tensor)
            return torch.argmax(q_values).item()

    def flatten_state(self, state):
        state_vector = []
        for key in state.keys():
            if isinstance(state[key], np.ndarray):
                state_vector.extend(state[key].flatten())
            elif isinstance(state[key], int):
                state_vector.append(state[key])
            elif isinstance(state[key], float):
                state_vector.append(state[key])
            else:
                raise ValueError(f"Unsupported state type for key {key}")

        return state_vector

    def update(self, state, action, next_state, reward, done):
        self.replay_buffer.push((state, action, next_state, reward, done))

        if len(self.replay_buffer.memory) > 32:
            states, actions, next_states, rewards, dones = self.replay_buffer.sample(32)

            # Convert states to flat vectors
            states = [self.flatten_state(s) for s in states]
            next_states = [self.flatten_state(s) for s in next_states]

            states = torch.FloatTensor(states)
            actions = torch.LongTensor(actions)
            next_states = torch.FloatTensor(next_states)
            rewards = torch.FloatTensor(rewards)
            dones = torch.FloatTensor(dones)

            q_values = self.q_network(states).gather(1, actions.unsqueeze(1))
            next_q_values = self.target_network(next_states).max(1)[0].detach()
            expected_q_values = rewards + self.gamma * next_q_values * (1 - dones)

            loss = nn.functional.mse_loss(q_values, expected_q_values.unsqueeze(1))
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            if done:
                self.target_network.load_state_dict(self.q_network.state_dict())


# Training loop
def train_dqn_agent(env, agent, num_episodes=Consts.NUM_EPISODES):
    for episode in range(num_episodes):
        if episode.numerator % 100:
            print(episode.numerator)

        state, info = env.reset()
        done = False

        reward = 0

        while not done:
            action = agent.select_action(state)
            next_state, reward, done, truncated, _ = env.step(action)
            agent.update(state, action, next_state, reward, done)
            state = next_state

            env.render()  # You might need to adjust this depending on your environment

        # print(reward)

        # if episode.numerator == num_episodes - 1:
        #     while True:
        #         action = agent.select_action(state)
        #         next_state, reward, done, _ = env.step(action)
        #         env.render()  # You might need to adjust this depending on your environment



def main():
    # window = arcade.Window(Consts.SCREEN_WIDTH, Consts.SCREEN_HEIGHT, Consts.SCREEN_TITLE)
    # menu_view = UI.MainMenu()

    # game_view = MyGame()
    # window.show_view(game_view)
    # arcade.run()

    # env = gym.make_vec('CustomGame-v0', 1)

    env = gym.make('CustomGame-v0')

    mode = "learn"
    input_size = 3
    output_size = 5

    # Initialize DQN agent
    agent = DQNAgent(input_size, output_size)

    if mode == "learn":
        # Train the DQN agent
        train_dqn_agent(env, agent)

        # agent.

        torch.save(agent.q_network.state_dict(), 'models/trained_model.pth')
    elif mode == "show":
        # Load the trained model
        loaded_model = DQN(input_size, output_size)
        loaded_model.load_state_dict(torch.load('models/trained_model.pth'))
        loaded_model.eval()  # Set the model to evaluation mode

        # Use the loaded model to make predictions
        state, info = env.reset()
        done = False

        while not done:
            # Assuming that the action space is discrete
            action = loaded_model(torch.FloatTensor(agent.flatten_state(state))).argmax().item()
            next_state, reward, done, truncated, _ = env.step(action)
            state = next_state
            env.render()

        env.close()

    # # env.set_window(window)
    # env.set_epsilon(0.2)
    # observation, info = env.reset()
    # done = False
    # do = 1
    # print(info["distance"])
    # # arcade.run()
    #
    # i = 0
    #
    # while not done:
    #     start_time = time.perf_counter()
    #     action = env.get_action(do)  # Replace with your RL agent's action
    #     prev_info = info["distance"]
    #     observation, reward, done, info = env.step(action)
    #     # print(reward)
    #     env.render()
    #     end_time = time.perf_counter()
    #
    #     frame_time = abs(0.016 - (end_time - start_time))
    #     # print(frame_time)
    #     # time.sleep(frame_time)
    #     # print(info["distance"])
    #
    #     if info["distance"] < prev_info:
    #         do = 5
    #     else:
    #         if i % 2:
    #             do = 2
    #         else:
    #             do = 4
    #
    #         i += 1
    #         # print(i)
    #
    #     if i > 1000:
    #         observation, info = env.reset()
    #         i = 0
    #
    #     if observation["end"]:
    #         observation, info = env.reset()
    #         i = 0
    #
    # env.close()


if __name__ == "__main__":
    main()
