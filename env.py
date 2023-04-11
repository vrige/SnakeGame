# our environement here is adapted from: https://github.com/TheAILearner/Snake-Game-using-OpenCV-Python/blob/master/snake_game_using_opencv.ipynb
import gym
from gym import spaces
import numpy as np
import cv2
import random
import time
from collections import deque

def collision_with_apple(apple_position, score):
    apple_position = [random.randrange(1, 50) * 10, random.randrange(1, 50) * 10]
    score += 1
    return apple_position, score

def collision_with_boundaries(snake_head):
    if snake_head[0] >= 500 or snake_head[0] < 0 or snake_head[1] >= 500 or snake_head[1] < 0:
        return 1
    else:
        return 0

def collision_with_self(snake_position):
    snake_head = snake_position[0]
    if snake_head in snake_position[1:]:
        return 1
    else:
        return 0

class SneakEnv(gym.Env):
    def __init__(self, reward_system=None, rending=False, snake_len_goal=30):
        super(SneakEnv, self).__init__()

        # reward system can be passed from outside the class.
        # It must be a dict and it must contain the following fields:
        # apple, nothing, die and tot_rew.
        # Because they are linked with the events of the game where we want the snake to learn.
        self.reward_system = reward_system
        if reward_system is None:
            self.reward_system = {
                "apple": 10000,
                "nothing": 0,
                "die": -1000,
                "dist": True,
                "dist_f": lambda x, y: np.linalg.norm(x-y),
                "tot_rew": lambda x, y, z: (250 - self.reward_system["dist_f"](x, y) + z) / 100
                            if (self.reward_system["dist"]) else (250 + z) / 100
            }

        # The goal of the snake is to reach a certain length
        self.snake_len_goal = snake_len_goal

        self.rending = rending
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
        self.render()
        button_direction = action
        # Change the head position based on the button direction
        if button_direction == 1:
            self.snake_head[0] += 10
        elif button_direction == 0:
            self.snake_head[0] -= 10
        elif button_direction == 2:
            self.snake_head[1] += 10
        elif button_direction == 3:
            self.snake_head[1] -= 10

        event_reward = 0
        # Increase Snake length on eating apple
        if self.snake_head == self.apple_position:
            self.apple_position, self.score = collision_with_apple(self.apple_position, self.score)
            self.snake_position.insert(0, list(self.snake_head))
            event_reward = self.reward_system["apple"]

        else:
            self.snake_position.insert(0, list(self.snake_head))
            self.snake_position.pop()
            event_reward = self.reward_system["nothing"]

        # On collision kill the snake and print the score
        if collision_with_boundaries(self.snake_head) == 1 or collision_with_self(self.snake_position) == 1:
            self.collision = True
            self.render()
            self.done = True
            event_reward = self.reward_system["die"]

        self.total_reward = self.reward_system["tot_rew"](np.array(self.snake_head),
                                                          np.array(self.apple_position), event_reward)

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
        self.img = np.zeros((500, 500, 3), dtype='uint8')
        # Initial Snake and Apple position
        self.snake_position = [[250, 250], [240, 250], [230, 250]]
        self.apple_position = [random.randrange(1, 50) * 10, random.randrange(1, 50) * 10]
        self.score = 0
        self.prev_button_direction = 1
        self.button_direction = 1
        self.snake_head = [250, 250]

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
                self.img = np.zeros((500, 500, 3), dtype='uint8')
                # Display Apple
                cv2.rectangle(self.img, (self.apple_position[0], self.apple_position[1]),
                              (self.apple_position[0] + 10, self.apple_position[1] + 10), (0, 0, 255), 3)
                # Display Snake
                for position in self.snake_position:
                    cv2.rectangle(self.img, (position[0], position[1]), (position[0] + 10, position[1] + 10), (0, 255, 0),
                                  3)

                # Takes step after fixed time
                t_end = time.time() + 0.05
                k = -1
                while time.time() < t_end:
                    if k == -1:
                        k = cv2.waitKey(1)
                    else:
                        continue
            else:
                font = cv2.FONT_HERSHEY_SIMPLEX
                self.img = np.zeros((500, 500, 3), dtype='uint8')
                cv2.putText(self.img, 'Your Score is {}'.format(self.score), (140, 250), font,
                            1, (255, 255, 255),2,cv2.LINE_AA)
                cv2.imshow('snake', self.img)

