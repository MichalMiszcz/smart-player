from collections import defaultdict
from typing import Tuple

import gym
import numpy as np
from gym import spaces
import arcade
import math

from gym.core import ActType
from gym.core import ObsType

import Player
import Enemy
import UI
import Consts


class MyGame(arcade.Window):

    def __init__(self, width, height, title):

        super().__init__(width, height, title)

        # self.set_update_rate(1/60)

        self.end = False
        self.time = 0
        self.is_done = False
        self.max_x = 0
        self.tile_map = None
        self.scene = None
        self.player_sprite = None
        self.physics_engine = None
        self.player_speed = 0
        self.player_acc = 0
        self.down_pressed = False
        self.bought = False
        self.health = 10
        self.died = False

        # Shooting mechanics
        self.can_shoot = False
        self.shoot_timer = 0
        self.shoot_pressed = False
        self.bullets = 0

        # UI and camera
        self.camera = None
        self.gui_camera = None
        self.score = 0
        self.old_score = 0
        self.reset_score = True
        self.end_of_map = 0
        self.end_of_map1 = 0
        self.level = 1

        # Sounds
        self.collect_coin_sound = arcade.load_sound(":resources:sounds/coin1.wav")
        self.jump_sound = arcade.load_sound(":resources:sounds/jump1.wav")
        self.lose = arcade.load_sound(":resources:sounds/lose5.wav")
        self.game_over = arcade.load_sound(":resources:sounds/gameover4.wav")
        self.shoot_sound = arcade.load_sound(":resources:sounds/hurt5.wav")
        self.hit_sound = arcade.load_sound(":resources:sounds/hit5.wav")
        self.upgrade = arcade.load_sound(":resources:sounds/upgrade1.wav")

        self.new_enemy = 0

        self.previous_coin_distance = 10000
        self.previous_spike_distance = 10000
        self.previous_position = 128

        arcade.set_background_color(arcade.csscolor.CORNFLOWER_BLUE)

    def setup(self):
        self.level = 1
        self.camera = arcade.Camera(self.width, self.height)
        self.gui_camera = arcade.Camera(self.width, self.height)

        self.is_done = False

        self.previous_coin_distance = 10000

        if self.level <= 1:
            map_name = f"levels/Level{self.level}.json"
        else:
            # game_won = UI.GameWonView()
            # self.window.show_view(game_won)
            print("You won")
            self.end = True


        self.new_enemy = 0

        # Layer Specific Options for the Tilemap
        layer_options = {
            Consts.LAYER_NAME_PLATFORMS: {
                "use_spatial_hash": True,
            },
            Consts.LAYER_NAME_COINS: {
                "use_spatial_hash": True,
            },
            Consts.LAYER_NAME_DONT_TOUCH: {
                "use_spatial_hash": True,
            },
            "Sklep": {
                "use_spatial_hash": True,
            },
        }

        self.tile_map = arcade.load_tilemap(map_name, Consts.TILE_SCALING, layer_options)
        self.scene = arcade.Scene.from_tilemap(self.tile_map)

        if self.reset_score:
            self.score = 0
        self.reset_score = True

        # Shooting mechanics
        self.can_shoot = True
        self.shoot_timer = 0

        self.player_sprite = Player.PlayerCharacter()
        self.player_sprite.center_x = Consts.PLAYER_START_X
        self.player_sprite.center_y = Consts.PLAYER_START_Y
        self.scene.add_sprite(Consts.LAYER_NAME_PLAYER, self.player_sprite)

        # self.max_x = Consts.PLAYER_START_X

        # variable for detecting if player ended the level
        self.end_of_map = (self.tile_map.width - 5) * Consts.GRID_PIXEL_SIZE
        # variable for detecting right edge of tilemap
        self.end_of_map1 = (self.tile_map.width - 16) * Consts.GRID_PIXEL_SIZE

        # Enemies
        enemies_layer = self.tile_map.object_lists[Consts.LAYER_NAME_ENEMIES]

        for my_object in enemies_layer:
            cartesian = self.tile_map.get_cartesian(
                my_object.shape[0], my_object.shape[1]
            )
            enemy = Enemy.Enemy()

            enemy.center_x = math.floor(
                cartesian[0] * Consts.TILE_SCALING * self.tile_map.tile_width
            )
            enemy.center_y = math.floor(
                (cartesian[1] + 1) * (self.tile_map.tile_height * Consts.TILE_SCALING)
            )

            enemy.physics_engine = arcade.PhysicsEnginePlatformer(
                enemy,
                gravity_constant=Consts.GRAVITY,
                walls=self.scene[Consts.LAYER_NAME_PLATFORMS],
            )
            self.scene.add_sprite(Consts.LAYER_NAME_ENEMIES, enemy)

        self.scene.add_sprite_list(Consts.LAYER_NAME_BULLETS)

        if self.tile_map.background_color:
            arcade.set_background_color(self.tile_map.background_color)

        # Create the physics engine
        self.physics_engine = arcade.PhysicsEnginePlatformer(
            self.player_sprite,
            gravity_constant=Consts.GRAVITY,
            walls=self.scene[Consts.LAYER_NAME_PLATFORMS],
        )

    def death(self):
        self.health -= 1

        if self.health > 0:
            # don't reset game
            return 1
        else:
            # reset game
            self.health = 5
            return 0

    # def on_show_view(self):
    #     self.setup()

    def on_draw(self):
        arcade.start_render()

        # Clear the screen to the background color
        self.clear()
        self.camera.use()

        # Draw scene
        self.scene.draw()

        self.gui_camera.use()

        score_text = f"Score: {self.score}"
        arcade.draw_text(
            score_text,
            10,
            10,
            arcade.csscolor.WHITE,
            18,
        )

        life_text = f"Health: {self.health}"
        arcade.draw_text(
            life_text,
            10,
            630,
            arcade.csscolor.RED,
            18,
        )

        bullet_text = f"Bullets: {self.bullets}"
        arcade.draw_text(
            bullet_text,
            870,
            10,
            arcade.csscolor.WHITE,
            18,
        )

    def take_action(self, action):
        # action 1: go left
        # action 2: jump left
        # action 3: jump
        # action 4: jump right
        # action 5: go right
        if action == 1:
            self.player_speed = -1
        elif action == 2:
            self.player_speed = -1
            if self.physics_engine.can_jump():
                self.player_sprite.change_y = Consts.PLAYER_JUMP_SPEED
                # arcade.play_sound(self.jump_sound)
        elif action == 3:
            if self.physics_engine.can_jump():
                self.player_sprite.change_y = Consts.PLAYER_JUMP_SPEED
                # arcade.play_sound(self.jump_sound)
        elif action == 4:
            self.player_speed = 1
            if self.physics_engine.can_jump():
                self.player_sprite.change_y = Consts.PLAYER_JUMP_SPEED
                # arcade.play_sound(self.jump_sound)
        elif action == 5:
            self.player_speed = 1

    def center_camera_to_player(self):
        screen_center_x = self.player_sprite.center_x - (self.camera.viewport_width / 2)
        screen_center_y = self.player_sprite.center_y - (
                self.camera.viewport_height / 2
        )

        if screen_center_x < 0:
            screen_center_x = 0
        if screen_center_y < 0:
            screen_center_y = 0
        if screen_center_x > self.end_of_map1:
            screen_center_x = self.end_of_map1
        player_centered = screen_center_x, screen_center_y

        self.camera.move_to(player_centered)

    def players_death(self):
        self.died = True

        self.is_done = True

        # if self.death() == 1:
        #     self.setup()
        #     # self.player_sprite.change_x = 0
        #     # self.player_sprite.change_y = 0
        #     # self.player_sprite.center_x = Consts.PLAYER_START_X
        #     # self.player_sprite.center_y = Consts.PLAYER_START_Y
        #     # arcade.play_sound(self.lose)
        # else:
        #     self.level = 1
        #     # game_over = UI.GameOverView()
        #     # self.window.show_view(game_over)
        #     # arcade.play_sound(self.game_over)

    def on_update(self, delta_time):

        self.time += 1
        # print(self.time)
        # if self.time > Consts.RESET_TIME:
        #     self.reset = True

        # Enemies
        self.new_enemy += 1

        # if self.new_enemy == 240:
        #     enemies_layer = self.tile_map.object_lists[Consts.LAYER_NAME_ENEMIES]
        #
        #     for my_object in enemies_layer:
        #         cartesian = self.tile_map.get_cartesian(
        #             my_object.shape[0], my_object.shape[1]
        #         )
        #         enemy = Enemy.Enemy()
        #
        #         enemy.center_x = math.floor(
        #             cartesian[0] * Consts.TILE_SCALING * self.tile_map.tile_width
        #         )
        #         enemy.center_y = math.floor(
        #             (cartesian[1] + 1) * (self.tile_map.tile_height * Consts.TILE_SCALING)
        #         )
        #
        #         enemy.physics_engine = arcade.PhysicsEnginePlatformer(
        #             enemy,
        #             gravity_constant=Consts.GRAVITY,
        #             walls=self.scene[Consts.LAYER_NAME_PLATFORMS],
        #         )
        #         self.scene.add_sprite(Consts.LAYER_NAME_ENEMIES, enemy)
        #
        #         self.new_enemy = 0

        # Move the player with the physics engine
        self.physics_engine.update()

        if self.physics_engine.can_jump():
            if self.player_speed > 0:
                if self.player_sprite.change_x < Consts.PLAYER_MOVEMENT_SPEED_MAX:
                    self.player_sprite.change_x += Consts.PLAYER_MOVEMENT_ACCELERATION * self.player_speed
            elif self.player_speed < 0:
                if self.player_sprite.change_x > -Consts.PLAYER_MOVEMENT_SPEED_MAX:
                    self.player_sprite.change_x += Consts.PLAYER_MOVEMENT_ACCELERATION * self.player_speed
            else:
                self.player_sprite.change_x = 0

        if self.bullets == 0:
            self.can_shoot = False

        if self.can_shoot:
            if self.shoot_pressed:
                # arcade.play_sound(self.shoot_sound)
                bullet = arcade.Sprite("levels/Sprites/pocisk.png", 1)

                if self.player_sprite.character_face_direction == Consts.RIGHT_FACING:
                    bullet.change_x = Consts.BULLET_SPEED
                else:
                    bullet.change_x = -Consts.BULLET_SPEED

                bullet.center_x = self.player_sprite.center_x
                bullet.center_y = self.player_sprite.center_y

                self.scene.add_sprite(Consts.LAYER_NAME_BULLETS, bullet)
                self.bullets -= 1
                self.can_shoot = False
        else:
            self.shoot_timer += 1
            if self.shoot_timer == Consts.SHOOT_SPEED:
                self.can_shoot = True
                self.shoot_timer = 0

        # Animations
        if self.physics_engine.can_jump():
            self.player_sprite.can_jump = False
        else:
            self.player_sprite.can_jump = True

        if self.level == 1:
            self.scene.update_animation(
                delta_time, [Consts.LAYER_NAME_COINS, Consts.LAYER_NAME_PLAYER, "Znaki", Consts.LAYER_NAME_ENEMIES]
            )
        else:
            self.scene.update_animation(
                delta_time, [Consts.LAYER_NAME_COINS, Consts.LAYER_NAME_PLAYER, "Znaki",
                             Consts.LAYER_NAME_ENEMIES, "Sklep"]
            )

        self.scene.update([Consts.LAYER_NAME_ENEMIES, Consts.LAYER_NAME_BULLETS])

        for enemy in self.scene[Consts.LAYER_NAME_ENEMIES]:
            enemy.physics_engine.update()

            if enemy.physics_engine.can_jump():
                enemy.change_x = enemy.move_speed

            if arcade.check_for_collision_with_list(
                    enemy, self.scene[Consts.LAYER_NAME_DONT_TOUCH]
            ):
                enemy.remove_from_sprite_lists()

            if enemy.center_y < -100:
                enemy.remove_from_sprite_lists()

        for bullet in self.scene[Consts.LAYER_NAME_BULLETS]:
            hit_list = arcade.check_for_collision_with_lists(
                bullet,
                [
                    self.scene[Consts.LAYER_NAME_ENEMIES],
                    self.scene[Consts.LAYER_NAME_PLATFORMS],
                ],
            )

            if hit_list:
                bullet.remove_from_sprite_lists()

                for collision in hit_list:
                    if (
                            self.scene[Consts.LAYER_NAME_ENEMIES]
                            in collision.sprite_lists
                    ):
                        collision.death()
                        # arcade.play_sound(self.hit_sound)

                return

            if (bullet.right < 0) or (
                    bullet.left
                    > (self.tile_map.width * self.tile_map.tile_width) * Consts.TILE_SCALING
            ):
                bullet.remove_from_sprite_lists()

        player_collision_list = arcade.check_for_collision_with_lists(
            self.player_sprite,
            [
                self.scene[Consts.LAYER_NAME_COINS],
                self.scene[Consts.LAYER_NAME_ENEMIES],
            ],
        )

        # Loop through each coin we hit (if any) and remove it
        for collision in player_collision_list:
            if self.scene[Consts.LAYER_NAME_ENEMIES] in collision.sprite_lists:
                # self.players_death()
                return
            else:
                collision.remove_from_sprite_lists()
                # Play a sound
                # arcade.play_sound(self.collect_coin_sound)
                self.score += 1

        if self.player_sprite.center_y < -100:
            self.players_death()

        # Did the player touch something they should not?
        if arcade.check_for_collision_with_list(
                self.player_sprite, self.scene[Consts.LAYER_NAME_DONT_TOUCH]
        ):
            self.players_death()

        # Checking if player touched shop
        if self.level > 1:
            if arcade.check_for_collision_with_list(
                    self.player_sprite, self.scene["Sklep"]
            ):
                if self.down_pressed and not self.bought:
                    if self.score >= 2:
                        self.bullets += 1
                        self.score -= 2
                        self.bought = True
                        # arcade.play_sound(self.upgrade)

        # See if the player got to the end of the level
        if self.player_sprite.center_x >= self.end_of_map:
            self.level += 1

            # Make sure to keep the score from this level when setting up the next level
            self.reset_score = False

            self.setup()

        self.center_camera_to_player()

    def get_something_positions(self, layer):
        coin_positions = []
        coin_layer = self.scene[layer]

        for coin_object in coin_layer:
            # Calculate the position of each coin and add it to the list
            cartesian = self.tile_map.get_cartesian(
                coin_object.center_x, coin_object.center_y
            )
            coin_position = math.floor(
                cartesian[0] * Consts.TILE_SCALING * self.tile_map.tile_width
            )
            coin_positions.append(coin_position)

        return coin_positions

    def calculate_nearest_something_distance(self, layer):
        agent_pos = self.player_sprite.center_x
        coin_positions = self.get_something_positions(layer)

        if not coin_positions:
            return float('inf')  # No coins present

        # Calculate the distance to the nearest coin
        nearest_coin_distance = min(abs(agent_pos - coin_pos) for coin_pos in coin_positions)

        return nearest_coin_distance

    def get_height(self):
        return self.tile_map.height * Consts.SPRITE_PIXEL_SIZE

    def get_width(self):
        return self.tile_map.height * Consts.SPRITE_PIXEL_SIZE

    def get_observation(self):
        agent_pos = self.player_sprite.center_x
        target_pos = self.end_of_map


        coin_positions = self.get_something_positions(Consts.LAYER_NAME_COINS)
        spikes_positions = self.get_something_positions(Consts.LAYER_NAME_DONT_TOUCH)

        coin_observation = [1 if agent_pos in coin_positions else 0]
        spikes_observation = [1 if agent_pos in spikes_positions else 0]
        return {
            "agent": agent_pos,
            "target": target_pos,
            "end": self.end,
            # "coins": coin_observation,  # Include coin information in the observation
            # "spikes": spikes_observation  # Include spikes information in the observation
        }

    def get_info(self):
        return {"info": False}


        # return {"distance": np.linalg.norm(self.player_sprite.center_x - self.end_of_map)}

    def get_reward(self):

        reward = 0

        if self.died:
            self.died = False
            reward += -2

        # if self.score > self.old_score:
        #     self.old_score = self.score
        #     reward += 200

        # print(self.player_sprite.center_x)
        if self.max_x > self.player_sprite.center_x:
            reward += -5
        elif self.max_x < self.player_sprite.center_x:
            self.max_x = self.player_sprite.center_x
            reward += 5

        if self.previous_position > self.player_sprite.center_x:
            reward += -5
        elif self.previous_position < self.player_sprite.center_x:
            reward += 15
        else:
            reward += -200

        if self.player_sprite.center_x >= self.end_of_map - 10:
            reward += 200

        # # Calculate the distance to the nearest coin
        # nearest_coin_distance = self.calculate_nearest_something_distance(Consts.LAYER_NAME_COINS)
        #
        # # Define reward components for moving towards/away from coins
        # reward_near_coin = 30  # Reward for getting closer to coins
        # reward_away_coin = -10  # Penalty for moving away from coins
        #
        # # Calculate the reward based on the change in distance to the nearest coin
        # delta_distance = self.previous_coin_distance - nearest_coin_distance
        #
        # # Determine if the agent is moving closer to or away from coins
        # if delta_distance > 0:
        #     reward += reward_near_coin
        # elif delta_distance == 0:
        #     reward += -20
        # else:
        #     reward += reward_away_coin
        #
        # # Update the previous coin distance for the next step
        # self.previous_coin_distance = nearest_coin_distance

        # # Calculate the distance to the nearest coin
        # nearest_spikes_distance = self.calculate_nearest_something_distance(Consts.LAYER_NAME_DONT_TOUCH)
        #
        # # Define reward components for moving towards/away from coins
        # reward_near_spikes = -1  # Reward for getting closer to coins
        # reward_away_spikes = 1  # Penalty for moving away from coins
        #
        # # Calculate the reward based on the change in distance to the nearest coin
        # delta_distance = self.previous_spike_distance - nearest_spikes_distance
        #
        # # Determine if the agent is moving closer to or away from coins
        # if abs(delta_distance) > 20:
        #     reward += reward_near_spikes
        # elif delta_distance == 0:
        #     reward += -1
        # else:
        #     reward += reward_away_spikes
        #
        # # Update the previous coin distance for the next step
        # self.previous_spike_distance = nearest_spikes_distance

        # print(self.previous_coin_distance)

        if self.end:
            reward += 1000

        return reward

    def done(self):
        return self.is_done


class CustomGameEnvironment(gym.Env):
    def __init__(self):
        super(CustomGameEnvironment, self).__init__()

        # Initialization of game
        self.game = MyGame(Consts.SCREEN_WIDTH, Consts.SCREEN_HEIGHT, "Title")
        self.game.setup()

        # Define observation and action spaces
        self.observation_space = spaces.Dict(
            {
                "agent": spaces.Box(0, 6144, shape=(2,), dtype=float),
                "target": spaces.Box(0, 6144, shape=(2,), dtype=float),
                "end": spaces.Discrete(2),
                # "coins": spaces.Box(0, 1, shape=(1,), dtype=int),  # Include coin information in the observation
                # "spikes": spaces.Box(0, 1, shape=(1,), dtype=int)  # Include spikes information in the observation
            }
        )

        self.action_space = gym.spaces.discrete.Discrete(Consts.NUMBER_OF_ACTIONS)

        # self.window = None

        # self.epsilon = 0.1
        self.q_values = defaultdict(lambda: np.zeros(5))
        self.max_iterations = 1000
        self.iterations = 0

    # def set_epsilon(self, epsilon):
    #     self.epsilon = epsilon

    # def set_window(self, window):
    #     self.window = window

    def step(self, action):
        # Take action in the game
        self.game.take_action(action)

        # Get observation, reward, done, info from the game
        observation = self.game.get_observation()
        reward = self.game.get_reward()
        done = self.game.done()
        info = self.game.get_info()  # Additional information, if needed

        self.iterations += 1
        # print(reward)

        if self.iterations > self.max_iterations:
            print("reset")
            self.game.is_done = True

        return observation, reward, done, info

    def reset(self, **kwargs):
        # Reset the game and return initial observation
        self.iterations = 0
        self.game.setup()
        # self.window(self.game)
        observation = self.game.get_observation()
        info = self.game.get_info()  # Additional information, if needed

        return observation, info

    def render(self, mode='human', ):
        self.game.dispatch_events()
        self.game._dispatch_updates(1)
        self.game.on_draw()  # Draw the frame
        self.game.flip()  # Display the frame
        # self.game.update(1/60)  # Update the game state

    def close(self):
        # Close any resources when done
        self.game.setup()

    # def get_action(self, do):
    #     if np.random.random() < self.epsilon:
    #         return np.random.randint(1, 6)
    #     else:
    #         return do
