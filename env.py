# our environement here is adapted from: https://github.com/TheAILearner/Snake-Game-using-OpenCV-Python/blob/master/snake_game_using_opencv.ipynb
import gym
from gym import spaces
import numpy as np
import cv2
import random
import time
from collections import deque

def collision_with_apple(score, dim, snake_position, cell_size):
    apple_in_snake = True
    while apple_in_snake:
        apple_position = [random.randrange(1, int(dim/cell_size)) * cell_size, random.randrange(1, int(dim/cell_size)) * cell_size]
        if apple_position in snake_position:
            apple_in_snake = True
        else:
            apple_in_snake = False
    score += 1
    return apple_position, score

def collision_with_boundaries(snake_head, dim):
    if snake_head[0] >= dim or snake_head[0] < 0 or snake_head[1] >= dim or snake_head[1] < 0:
        return 1
    else:
        return 0

def collision_with_self(snake_position):
    snake_head = snake_position[0]
    if snake_head in snake_position[1:]:
        return 1
    else:
        return 0

def getting_close(previous_head, snake_head, apple_position):
    if (abs(previous_head[0] - apple_position[0]) + abs(previous_head[1] - apple_position[1])) <= (abs(snake_head[0] - apple_position[0]) + abs(snake_head[1] - apple_position[1])):
        return False
    else:
        return True

def move_and_check_illegal_move(snake_position, move_direction, cell_size):
    new_snake_head = snake_position[0].copy()
    illegal = False

    if move_direction == 0:
        new_snake_head[0] -= cell_size   # left
    elif move_direction == 1:
        new_snake_head[1] -= cell_size   # up
    elif move_direction == 2:
        new_snake_head[0] += cell_size   # right
    elif move_direction == 3:
        new_snake_head[1] += cell_size   # down

    if new_snake_head in snake_position[1:2]:
        new_snake_head = snake_position[0].copy()
        illegal = True

    return new_snake_head, illegal

# Check that the dim is one of the following: 100, 200, 300, 400 or 500.
# In the case it is not, then take the default size: 500
def check_dim(dim):
    if (dim % 100) != 0 or dim > 500 or dim < 100:
        return 500
    return dim

# For visual reason, amplify the dim.
# - dim = 300 -> x2 -> new_dim = 600
# - dim = 200 -> x3 -> new_dim = 600
# - dim = 100 -> x5 -> new_dim = 500
def amplify_dim(dim):
    if dim == 300:
        return 2*dim, 10*2
    elif dim == 200:
        return 3*dim, 10*3
    elif dim == 100:
        return 5*dim, 10*5
    else:
        return dim, 10

def get_death_info(snake_position):
    print("---------------------------------------------------")
    for pos in snake_position:
        print("pos_x: " + str(pos[0]) + ", pos_y: " + str(pos[1]))

class SneakEnv(gym.Env):
    def __init__(self, reward_system=None, rending=False, snake_len_goal=30, dim=500, time_speed=0.05):
        super(SneakEnv, self).__init__()

        # reward system can be passed from outside the class.
        # It must be a dict and it must contain the following fields:
        # apple, nothing, die and tot_rew.
        # Because they are linked with the events of the game where we want the snake to learn.
        self.reward_system = reward_system
        if reward_system is None:

            self.reward_system = {
                "apple": 10000,
                "close": 0,
                "far": 0,
                "die": -1,
                "illegal": -1,
                "dist_f": lambda x, y: np.linalg.norm(x-y),
                "tot_rew": lambda x, y, z: (50 - self.reward_system["dist_f"](x, y) + z) / 100
            }
            '''
            self.reward_system = {
                "apple": 1,
                "close": 0.01,
                "far": -0.01,
                "die": -1,
                "time": -1,
                "after-tot": 100
            }
            '''
        # The goal of the snake is to reach a certain length
        self.snake_len_goal = snake_len_goal

        self.rending = rending
        self.dim = check_dim(int(dim))
        self.dim_ampl, self.cell_dim = amplify_dim(self.dim)
        self.t_time = time_speed
        self.collision = False

        # Define action and observation space
        # They must be gym.spaces objects
        # Example when using discrete actions:
        self.action_space = spaces.Discrete(4)
        # Example for using image as input (channel-first; channel-last also works):
        self.observation_space = spaces.Box(low=-500, high=500,
                                            shape=(5 + self.snake_len_goal,), dtype=np.float32)


    def step(self, action):
        self.prev_actions.append(action)
        button_direction = action
        # Change the head position based on the button direction
        self.snake_head, illegal = move_and_check_illegal_move(self.snake_position, button_direction, self.cell_dim)

        if not illegal:

            self.render()
            self.snake_position.insert(0, list(self.snake_head))
            event_reward = 0
            # Increase Snake length on eating apple
            if self.snake_head == self.apple_position:
                self.apple_position, self.score = collision_with_apple(self.score, self.dim_ampl, self.snake_position, self.cell_dim)
                event_reward = self.reward_system["apple"]

            else:
                self.snake_position.pop()
                if getting_close(self.previous_head, self.snake_head, self.apple_position):
                    event_reward = self.reward_system["close"]
                else:
                    event_reward = self.reward_system["far"]

            # On collision kill the snake and print the score
            if collision_with_boundaries(self.snake_head, self.dim_ampl) == 1 or collision_with_self(self.snake_position) == 1:
                self.collision = True
                self.render()
                self.done = True
                event_reward = self.reward_system["die"]

            self.total_reward = self.reward_system["tot_rew"](np.array(self.snake_head),
                                                                  np.array(self.apple_position), event_reward)
        else:
            self.total_reward = self.reward_system["illegal"]

        self.previous_head = self.snake_head

        head_x = self.snake_head[0]
        head_y = self.snake_head[1]

        snake_length = len(self.snake_position)
        apple_delta_x = self.apple_position[0] - head_x
        apple_delta_y = self.apple_position[1] - head_y

        # create observation:
        obs = [head_x, head_y, apple_delta_x, apple_delta_y, snake_length] + list(self.prev_actions)
        obs = np.array(obs)

        reward = self.total_reward
        info = {}
        return obs, reward, self.done, info

    def reset(self):
        self.img = np.zeros((self.dim_ampl, self.dim_ampl, 3), dtype='uint8')
        dim = int(self.dim_ampl/2)

        # Initial Snake and Apple position
        self.snake_position = [[dim, dim], [dim - self.cell_dim, dim], [dim - self.cell_dim*2, dim]]
        self.apple_position = [random.randrange(1, int(self.dim_ampl/self.cell_dim)) * self.cell_dim,
                               random.randrange(1, int(self.dim_ampl/self.cell_dim)) * self.cell_dim]
        self.score = 0

        self.snake_head = [dim, dim]
        self.previous_head = self.snake_head

        self.collision = False
        self.done = False

        head_x = self.snake_head[0]
        head_y = self.snake_head[1]

        snake_length = len(self.snake_position)
        apple_delta_x = self.apple_position[0] - head_x
        apple_delta_y = self.apple_position[1] - head_y

        self.prev_actions = deque(maxlen=self.snake_len_goal)  # however long we aspire the snake to be
        for i in range(self.snake_len_goal):
            self.prev_actions.append(-1)  # to create history

        # create observation:
        obs = [head_x, head_y, apple_delta_x, apple_delta_y, snake_length] + list(self.prev_actions)
        obs = np.array(obs)

        return obs

    def render(self, mode='nothing'):
        # the idea is to allow the rendering during the training when it is activated by the callback
        if mode == 'console':
            self.rending = False
        elif mode == 'human':
            self.rending = True

        if self.rending:
            if not self.collision:
                cv2.imshow('snake', self.img)
                cv2.waitKey(1)
                self.img = np.zeros((self.dim_ampl, self.dim_ampl, 3), dtype='uint8')
                # Display Apple
                cv2.rectangle(self.img, (self.apple_position[0], self.apple_position[1]),
                              (self.apple_position[0] + self.cell_dim, self.apple_position[1] + self.cell_dim),
                              (0, 0, 255), 3)
                # Display Snake's head
                pos_head = self.snake_position[0]
                cv2.rectangle(self.img, (pos_head[0], pos_head[1]),
                              (pos_head[0] + self.cell_dim, pos_head[1] + self.cell_dim),
                              (255, 255, 255), 3)
                # Display Snake
                for position in self.snake_position[1:]:
                    cv2.rectangle(self.img, (position[0], position[1]),
                                  (position[0] + self.cell_dim, position[1] + self.cell_dim),
                                  (0, 255, 0), 3)

                # Takes step after fixed time
                t_end = time.time() + self.t_time
                k = -1
                while time.time() < t_end:
                    if k == -1:
                        k = cv2.waitKey(1)
                    else:
                        continue
            else:
                font = cv2.FONT_HERSHEY_SIMPLEX
                self.img = np.zeros((self.dim_ampl, self.dim_ampl, 3), dtype='uint8')
                cv2.putText(self.img, 'Your Score is {}'.format(self.score), (140, 250), font,
                            1, (255, 255, 255), 2, cv2.LINE_AA)
                cv2.imshow('snake', self.img)

