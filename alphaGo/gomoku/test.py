# play_with_alphazero.py
#
# 和训练好的 AlphaZero 井字棋模型对战
#
# 运行:
# python play_with_alphazero.py
#
# 需要:
# pip install pygame torch gymnasium numpy

import math
import numpy as np
import pygame
import torch
import torch.nn as nn
import torch.nn.functional as F

import gymnasium as gym
from gymnasium import spaces


# =========================================================
# 环境
# =========================================================

class TicTacToeEnv(gym.Env):
    def __init__(self):
        super().__init__()

        self.board = np.zeros((3, 3), dtype=np.int8)
        self.current_player = 1

        self.action_space = spaces.Discrete(9)
        self.observation_space = spaces.Box(
            low=-1,
            high=1,
            shape=(3, 3),
            dtype=np.int8
        )

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.board[:] = 0
        self.current_player = 1

        return self._obs(), {}

    def _obs(self):
        return self.board * self.current_player

    def clone(self):
        env = TicTacToeEnv()
        env.board = self.board.copy()
        env.current_player = self.current_player
        return env

    def legal_actions(self):
        return [
            i for i in range(9)
            if self.board[i // 3, i % 3] == 0
        ]

    def step(self, action):
        r, c = divmod(action, 3)

        if self.board[r, c] != 0:
            return self._obs(), -1.0, True, False, {}

        self.board[r, c] = self.current_player

        winner = self._winner()

        if winner != 0:
            return self._obs(), 1.0, True, False, {}

        if len(self.legal_actions()) == 0:
            return self._obs(), 0.0, True, False, {}

        self.current_player *= -1

        return self._obs(), 0.0, False, False, {}

    def _winner(self):
        b = self.board

        lines = []

        lines.extend(list(b))
        lines.extend(list(b.T))

        lines.append(np.diag(b))
        lines.append(np.diag(np.fliplr(b)))

        for line in lines:
            s = np.sum(line)

            if s == 3:
                return 1

            if s == -3:
                return -1

        return 0


# =========================================================
# 网络
# =========================================================

class PolicyValueNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(9, 128),
            nn.ReLU(),

            nn.Linear(128, 128),
            nn.ReLU(),
        )

        self.policy_head = nn.Linear(128, 9)
        self.value_head = nn.Linear(128, 1)

    def forward(self, x):
        x = x.reshape(x.shape[0], -1).float()

        h = self.net(x)

        policy_logits = self.policy_head(h)
        value = torch.tanh(
            self.value_head(h)
        ).squeeze(-1)

        return policy_logits, value


# =========================================================
# MCTS
# =========================================================

class Node:
    def __init__(self, prior=0.0):
        self.prior = prior

        self.visit_count = 0
        self.value_sum = 0.0

        self.children = {}

    @property
    def value(self):
        if self.visit_count == 0:
            return 0.0

        return self.value_sum / self.visit_count


def evaluate(env, model, device):
    obs = torch.tensor(
        env._obs(),
        device=device
    ).unsqueeze(0)

    with torch.no_grad():
        logits, value = model(obs)

    logits = logits[0].cpu().numpy()

    legal = env.legal_actions()

    mask = np.full(9, -1e9)
    mask[legal] = 0

    logits = logits + mask

    probs = np.exp(logits - np.max(logits))
    probs = probs / probs.sum()

    return probs, float(value.item())


def select_child(node, c_puct=1.5):
    best_score = -1e9

    best_action = None
    best_child = None

    total_visits = max(1, node.visit_count)

    for action, child in node.children.items():

        q = -child.value

        u = (
            c_puct
            * child.prior
            * math.sqrt(total_visits)
            / (1 + child.visit_count)
        )

        score = q + u

        if score > best_score:
            best_score = score
            best_action = action
            best_child = child

    return best_action, best_child


def run_mcts(env, model, device, num_simulations=100):

    root = Node()

    probs, _ = evaluate(env, model, device)

    for a in env.legal_actions():
        root.children[a] = Node(prior=probs[a])

    for _ in range(num_simulations):

        sim_env = env.clone()

        node = root
        path = [node]

        done = False

        while node.children:

            action, node = select_child(node)

            _, reward, terminated, truncated, _ = sim_env.step(action)

            done = terminated or truncated

            path.append(node)

            if done:
                break

        if done:
            value = reward

        else:
            probs, value = evaluate(sim_env, model, device)

            for a in sim_env.legal_actions():
                node.children[a] = Node(prior=probs[a])

        for n in reversed(path):
            n.visit_count += 1
            n.value_sum += value
            value = -value

    visits = np.zeros(9)

    for a, child in root.children.items():
        visits[a] = child.visit_count

    visits = visits / visits.sum()

    return visits


def ai_move(env, model, device):
    probs = run_mcts(
        env,
        model,
        device,
        num_simulations=200
    )

    action = np.argmax(probs)

    return action


# =========================================================
# pygame UI
# =========================================================

pygame.init()

SIZE = 600
CELL = SIZE // 3

screen = pygame.display.set_mode((SIZE, SIZE))
pygame.display.set_caption("AlphaZero TicTacToe")

font = pygame.font.SysFont(None, 80)

WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
BLUE = (80, 80, 255)
RED = (255, 80, 80)


def draw_board(board):

    screen.fill(WHITE)

    # 网格
    for i in range(1, 3):
        pygame.draw.line(
            screen,
            BLACK,
            (0, i * CELL),
            (SIZE, i * CELL),
            4
        )

        pygame.draw.line(
            screen,
            BLACK,
            (i * CELL, 0),
            (i * CELL, SIZE),
            4
        )

    # 棋子
    for r in range(3):
        for c in range(3):

            x = c * CELL + CELL // 2
            y = r * CELL + CELL // 2

            if board[r, c] == 1:
                text = font.render("X", True, BLUE)
                rect = text.get_rect(center=(x, y))
                screen.blit(text, rect)

            elif board[r, c] == -1:
                text = font.render("O", True, RED)
                rect = text.get_rect(center=(x, y))
                screen.blit(text, rect)

    pygame.display.flip()


# =========================================================
# 主程序
# =========================================================

device = "cuda" if torch.cuda.is_available() else "cpu"

model = PolicyValueNet().to(device)

# 加载参数
model.load_state_dict(
    torch.load(
        "alphazero_tictactoe.pt",
        map_location=device
    )
)

model.eval()

env = TicTacToeEnv()

obs, _ = env.reset()

human_player = 1
done = False

clock = pygame.time.Clock()

while True:

    draw_board(env.board)

    if done:
        pygame.time.wait(3000)
        break

    if env.current_player == human_player:

        for event in pygame.event.get():

            if event.type == pygame.QUIT:
                pygame.quit()
                exit()

            if event.type == pygame.MOUSEBUTTONDOWN:

                mx, my = pygame.mouse.get_pos()

                c = mx // CELL
                r = my // CELL

                action = r * 3 + c

                if action in env.legal_actions():

                    _, reward, terminated, truncated, _ = env.step(action)

                    done = terminated or truncated

                    if done:

                        draw_board(env.board)

                        winner = env._winner()

                        if winner == human_player:
                            print("你赢了")

                        elif winner == 0:
                            print("平局")

                        else:
                            print("AI赢了")

    else:
        pygame.time.wait(300)

        action = ai_move(env, model, device)

        _, reward, terminated, truncated, _ = env.step(action)

        done = terminated or truncated

        if done:

            draw_board(env.board)

            winner = env._winner()

            if winner == human_player:
                print("你赢了")

            elif winner == 0:
                print("平局")

            else:
                print("AI赢了")

    clock.tick(60)