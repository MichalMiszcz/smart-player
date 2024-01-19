import arcade
import Consts


def load_texture_pair(filename):
    return [
        arcade.load_texture(filename),
        arcade.load_texture(filename, flipped_horizontally=True),
    ]


class PlayerCharacter(arcade.Sprite):
    """Player Sprite"""

    def __init__(self):

        super().__init__()

        # Default to face-right
        self.character_face_direction = Consts.RIGHT_FACING

        # Used for flipping between image sequences
        self.cur_texture = 0
        self.delta = 0
        self.scale = Consts.CHARACTER_SCALING

        # Track our state
        self.jumping = False

        # --- Load Textures ---
        main_path = "images/player1/character1"

        # Load textures for idle standing
        self.idle_texture_pair = load_texture_pair(f"{main_path}0.png")
        self.jump_texture_pair = load_texture_pair(f"{main_path}1.png")
        self.fall_texture_pair = load_texture_pair(f"{main_path}2.png")

        # Load textures for walking
        self.walk_textures = []
        texture = load_texture_pair(f"{main_path}6.png")
        self.walk_textures.append(texture)
        texture = load_texture_pair(f"{main_path}7.png")
        self.walk_textures.append(texture)
        texture = load_texture_pair(f"{main_path}8.png")
        self.walk_textures.append(texture)
        texture = load_texture_pair(f"{main_path}9.png")
        self.walk_textures.append(texture)

        # Set the initial texture
        self.texture = self.idle_texture_pair[0]

        self.hit_box = self.texture.hit_box_points

    def update_animation(self, delta_time: float = 1 / 60):

        # Figure out if we need to flip face left or right
        if self.change_x < 0 and self.character_face_direction == Consts.RIGHT_FACING:
            self.character_face_direction = Consts.LEFT_FACING
        elif self.change_x > 0 and self.character_face_direction == Consts.LEFT_FACING:
            self.character_face_direction = Consts.RIGHT_FACING

        # Jumping animation
        if self.change_y > 0:
            self.texture = self.jump_texture_pair[self.character_face_direction]
            return
        elif self.change_y < 0:
            self.texture = self.fall_texture_pair[self.character_face_direction]
            return

        # Idle animation
        if self.change_x == 0:
            self.texture = self.idle_texture_pair[self.character_face_direction]
            return

        # Walking animation
        self.delta += 1
        if self.cur_texture == 1 or self.cur_texture == 3:
            if self.delta == 5:
                self.cur_texture += 1
                self.delta = 0
        else:
            if self.delta == 10:
                self.cur_texture += 1
                self.delta = 0

        if self.cur_texture > 3:
            self.cur_texture = 0

        self.texture = self.walk_textures[self.cur_texture][
            self.character_face_direction
        ]
