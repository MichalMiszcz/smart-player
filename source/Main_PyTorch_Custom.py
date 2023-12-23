import random
import time
import math
from collections import deque

import gymnasium as gym
from gymnasium.vector.utils import spaces

import Consts
import CustomGameEnvironment

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torch.nn.functional as F

gym.envs.register(
    id='CustomGame-v0',
    entry_point='CustomGameEnvironment:CustomGameEnvironment',
)

BATCH_SIZE = 128
GAMMA = 0.99
EPS_START = 0.75
EPS_END = 0.05
EPS_DECAY = Consts.MAX_ITERATIONS * Consts.NUM_EPISODES * 0.25
TAU = 0.005
LR = 0.01


# Define the Deep Q Network (DQN) model
class DQN(nn.Module):
    def __init__(self, input_size, output_size):
        super(DQN, self).__init__()
        self.layer01 = nn.Linear(input_size, 128)
        self.layer02 = nn.Linear(128, 128)
        self.layer03 = nn.Linear(128, output_size)

    def forward(self, x):
        x = F.relu(self.layer01(x))
        x = F.relu(self.layer02(x))
        return self.layer03(x)


# Replay Buffer to store experiences for training
class ReplayBuffer(object):
    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, transition):
        self.memory.append(transition)

    def sample(self, batch_size):
        return zip(*random.sample(self.memory, batch_size))

    def __len__(self):
        return len(self.memory)



# DQN Agent
class DQNAgent:
    def __init__(self, input_size, output_size, q_network, learning_rate=LR, gamma=GAMMA, epsilon=EPS_START):
        self.q_network = q_network
        self.target_network = DQN(input_size, output_size)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()
        self.optimizer = optim.AdamW(self.q_network.parameters(), verbose=1, lr=learning_rate, amsgrad=True)
        self.gamma = gamma
        self.epsilon = epsilon
        self.replay_buffer = ReplayBuffer(capacity=10000)
        self.steps_done = 0
        self.ending = False

    def select_action(self, state):
        eps_threshold = EPS_END + (EPS_START - EPS_END) * \
                        math.exp(-1. * self.steps_done / EPS_DECAY)
        self.steps_done += 1
        # print(eps_threshold)

        state_vector = self.flatten_state(state)
        state_tensor = torch.FloatTensor(state_vector)

        if not self.ending:
            if np.random.rand() < eps_threshold:
                return np.random.choice(Consts.NUMBER_OF_ACTIONS)

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

        if len(self.replay_buffer.memory) > BATCH_SIZE:
            states, actions, next_states, rewards, dones = self.replay_buffer.sample(BATCH_SIZE)

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

    def set_ending(self):
        self.ending = True


# Training loop
def train_dqn_agent(env, agent, num_episodes=Consts.NUM_EPISODES):
    for episode in range(num_episodes):
        # if episode.numerator % 100 == 1:
        #     print(episode.numerator)

        if episode.numerator > num_episodes - 3:
            agent.set_ending()

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

        if state["end"]:
            print("Zapisuję koniec")
            torch.save(agent.q_network.state_dict(), 'models/trained_model_end.pth')

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

    # if GPU is to be used
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(device)

    env = gym.make('CustomGame-v0')

    input_size = 5
    output_size = 5

    if Consts.LOAD or not Consts.LEARN:
        print("load")
        q_network = DQN(input_size, output_size)
        q_network.load_state_dict(torch.load('models/trained_model6.pth'))
        q_network.eval()  # Set the model to evaluation mode
    else:
        q_network = DQN(input_size, output_size)

    # Initialize DQN agent
    agent = DQNAgent(input_size, output_size, q_network)

    if Consts.LEARN:
        # Train the DQN agent
        train_dqn_agent(env, agent)

        torch.save(agent.q_network.state_dict(), 'models/trained_model6.pth')
    elif not Consts.LEARN:
        # # Initialize DQN agent
        # agent = DQNAgent(input_size, output_size, epsilon=0)
        agent.set_ending()

        # Use the loaded model to make predictions
        state, info = env.reset()

        print(state)

        done = False

        while not done:
            # Assuming that the action space is discrete
            # action = loaded_model(torch.FloatTensor(agent.flatten_state(state))).argmax().item()
            state_vector = agent.flatten_state(state)
            state_tensor = torch.FloatTensor(state_vector)

            with torch.no_grad():
                q_values = agent.q_network(state_tensor)
                # print(q_values)
                action = torch.argmax(q_values).item()
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
