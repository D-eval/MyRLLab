"""
最速降线
失败
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
from stable_baselines3 import PPO
import math

import os
os.makedirs("tiny_save", exist_ok=True)
import matplotlib.pyplot as plt

save_fig_dir = "./tiny_save/fallcurve"
os.makedirs(save_fig_dir, exist_ok=True)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))
# 降落曲线
# action: 当前位置前进一格的同时下落几格
# Discrete(10), 0~9
# state: time, x, y, v

destination_position = (10, 10)
gravity = 10
l = 1

victory_reward = 10
failure_reward = -10
stage_reward = 10
stage_num = 10

total_distance = np.sqrt(destination_position[0]**2 + destination_position[1]**2)
stage_distance = total_distance / stage_num
# 多阶段学习
# 阶段一：学会走到终点，违反规则后直接done并给负reward
class FallCurve(gym.Env):
    def __init__(self):
        self.observation_space = spaces.Box(low=0,high=100,shape=(4,),dtype=np.float32)
        self.action_space = spaces.Box(
            low=0,
            high=math.pi/2,
            shape=(1,),
            dtype=np.float32
        )
        self.state = np.zeros(4)
        self.max_steps = 1000
    def reset(self, seed=None, options=None):
        self.step_count = 0
        self.state = np.array([0, 0, 0, 0])

        return self.state, {}

    def step(self, action):
        reward = 0

        self.step_count += 1
        truncated = False
        if self.step_count >= self.max_steps:
            truncated = True
        
        time, x, y, v = self.state

        theta = action
        dx = l * math.cos(theta)
        dy = l * math.sin(theta)
        
        x_new = x + dx
        y_new = y + dy
        
        # if x_rest < 0.1:
        #     x_new = destination_position[0]

        # if y_rest < 0.1:
        #     y_new = destination_position[1]
        
        done = False
        if max(destination_position[0]-x_new, destination_position[1]-y_new) <= 0.1:
            done = True
            reward += victory_reward
        else:
            if x_new > destination_position[0] or y_new > destination_position[1]:
                reward += failure_reward
                done = True

        # 阶段奖励
        rest_distance_old = np.sqrt((destination_position[0]-x)**2 + (destination_position[1]-y)**2)
        rest_distance_new = np.sqrt((destination_position[0]-x_new)**2 + (destination_position[1]-y_new)**2)
        reach_new_stage = (rest_distance_old // stage_distance) != (rest_distance_new // stage_distance)
        if reach_new_stage:
            reward += 5

        h = dy
        Delta_v_square = 2 * gravity * h
        v_new = np.sqrt(v ** 2 + Delta_v_square)
        
        distance = np.sqrt(dx**2 + dy**2)
        time_cost = 2 * distance / max(v + v_new, 1e-6)
        time_new = time + time_cost

        self.state = np.array([time_new, x_new, y_new, v_new])

        reward -= time_cost
        
        # print(reward, time_cost, x_new, y_new)
        if truncated:
            reward -= 10.0

        # old_dist = np.sqrt(
        #     (destination_position[0]-x)**2 +
        #     (destination_position[1]-y)**2
        # )

        # new_dist = np.sqrt(
        #     (destination_position[0]-x_new)**2 +
        #     (destination_position[1]-y_new)**2
        # )

        # reward += old_dist - new_dist
        
        return self.state, reward, done, truncated, {}


env = FallCurve()
obs, info = env.reset()

model = PPO(
    "MlpPolicy",
    env,
    verbose=1,
    device="cpu",
    n_steps=16,
    batch_size=16,
    learning_rate=1e-4,
)

total_times = []
for i in range(1000):

    model.learn(
        total_timesteps=1000,
        reset_num_timesteps=False
    )

    obs, info = env.reset()

    done = False

    xs = [0]
    ys = [0]

    actions = []

    while not done:

        action, _ = model.predict(
            obs,
            deterministic=True
        )

        actions.append(action.copy())

        obs, reward, done, _, _ = env.step(action)

        xs.append(obs[1])
        ys.append(obs[2])

    total_time = obs[0]
    total_times.append(total_time)
    
    plt.figure(figsize=(5,5))

    plt.plot(xs, ys, marker='o')

    plt.xlim(0, destination_position[0])
    plt.ylim(0, destination_position[1])

    plt.xlabel("x")
    plt.ylabel("y")

    plt.title(f"iter={i}, time={total_time}")

    plt.grid()

    plt.savefig(
        os.path.join(save_fig_dir, f"temp.png")
    )

    plt.close()

    print(f"iter={i}")
    # print(actions)
    
plt.plot(total_times)
plt.savefig(os.path.join(save_fig_dir, f"time_cost.png"))