import time

import arcade
import arcade.gui
import gymnasium as gym

import Consts
import CustomGameEnvironment

gym.envs.register(
    id='CustomGame-v0',
    entry_point='CustomGameEnvironment:CustomGameEnvironment',
)


def main():
    # window = arcade.Window(Consts.SCREEN_WIDTH, Consts.SCREEN_HEIGHT, Consts.SCREEN_TITLE)
    # menu_view = UI.MainMenu()

    # game_view = MyGame()
    # window.show_view(game_view)
    # arcade.run()

    env = gym.make('CustomGame-v0')
    # env.set_window(window)
    env.set_epsilon(0.5)
    info = env.reset()
    done = False
    do = 1
    print(info["distance"])
    # arcade.run()

    i = 0

    while not done:
        start_time = time.perf_counter()
        action = env.get_action(do)  # Replace with your RL agent's action
        prev_info = info["distance"]
        observation, reward, done, info = env.step(action)
        # print(reward)
        env.render()
        end_time = time.perf_counter()
        frame_time = abs(0.016 - (end_time - start_time))
        # print(frame_time)
        # time.sleep(frame_time)
        # print(info["distance"])

        if info["distance"] < prev_info:
            do = 5
        else:
            if i % 2:
                do = 2
            else:
                do = 4

            i += 1
            print(i)

        if i > 1000:
            info = env.reset()
            i = 0

    env.close()


if __name__ == "__main__":
    main()
