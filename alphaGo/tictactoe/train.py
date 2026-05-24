# alphazero_gymnasium_tictactoe.py
import math
import random
import numpy as np
import gymnasium as gym
from gymnasium import spaces

import torch
import torch.nn as nn
import torch.nn.functional as F


# =========================
# Gymnasium 环境：井字棋
# =========================

class TicTacToeEnv(gym.Env):
    def __init__(self):
        super().__init__()
        self.board = np.zeros((3, 3), dtype=np.int8)
        self.current_player = 1

        self.action_space = spaces.Discrete(9)
        self.observation_space = spaces.Box(
            low=-1, high=1, shape=(3, 3), dtype=np.int8
        )

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.board[:] = 0
        self.current_player = 1
        return self._obs(), {}

    def _obs(self):
        # 永远从当前玩家视角看棋盘
        return self.board * self.current_player

    def legal_actions(self):
        return [i for i in range(9) if self.board[i // 3, i % 3] == 0]

    def step(self, action):
        r, c = divmod(action, 3)

        if self.board[r, c] != 0:
            # 非法落子，直接输
            return self._obs(), -1.0, True, False, {}

        self.board[r, c] = self.current_player

        winner = self._winner()
        if winner != 0:
            return self._obs(), 1.0, True, False, {}

        if len(self.legal_actions()) == 0:
            return self._obs(), 0.0, True, False, {}

        self.current_player *= -1
        return self._obs(), 0.0, False, False, {}

    def clone(self):
        env = TicTacToeEnv()
        env.board = self.board.copy()
        env.current_player = self.current_player
        return env

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


# =========================
# Policy-Value 网络
# =========================

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
        # x: (B, 3, 3)
        x = x.reshape(x.shape[0], -1).float()
        h = self.net(x)

        policy_logits = self.policy_head(h)
        value = torch.tanh(self.value_head(h)).squeeze(-1)

        return policy_logits, value


# =========================
# MCTS
# =========================

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
    obs = torch.tensor(env._obs(), device=device).unsqueeze(0)
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
        u = c_puct * child.prior * math.sqrt(total_visits) / (1 + child.visit_count)
        score = q + u

        if score > best_score:
            best_score = score
            best_action = action
            best_child = child

    return best_action, best_child


def run_mcts(env, model, device, num_simulations=64):
    root = Node()

    probs, _ = evaluate(env, model, device)
    for a in env.legal_actions():
        root.children[a] = Node(prior=probs[a])

    for _ in range(num_simulations):
        sim_env = env.clone()
        node = root
        path = [node]

        done = False
        reward = 0.0

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

        # 反向传播，注意玩家视角交替，所以 value 要取负
        for n in reversed(path):
            n.visit_count += 1
            n.value_sum += value
            value = -value

    visits = np.zeros(9, dtype=np.float32)
    for a, child in root.children.items():
        visits[a] = child.visit_count

    policy = visits / visits.sum()
    return policy


# =========================
# 自博弈
# =========================

def self_play_game(model, device, num_simulations=64):
    env = TicTacToeEnv()
    obs, _ = env.reset()

    data = []
    players = []

    done = False

    while not done:
        policy = run_mcts(env, model, device, num_simulations)

        data.append((env._obs().copy(), policy.copy()))
        players.append(env.current_player)

        action = np.random.choice(9, p=policy)
        _, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

    winner = env._winner()

    train_data = []
    for obs, policy in data:
        # obs 已经是当前玩家视角，所以胜负也要转成当前玩家视角
        if winner == 0:
            z = 0.0
        else:
            # 因为 obs 是落子前当前玩家视角
            # winner 与当时玩家相同则 z=1，否则 z=-1
            idx = len(train_data)
            player = players[idx]
            z = 1.0 if winner == player else -1.0

        train_data.append((obs, policy, z))

    return train_data


# =========================
# 训练
# =========================

def train_step(model, optimizer, batch, device):
    obs, target_policy, target_value = zip(*batch)

    obs = torch.tensor(np.array(obs), device=device).float()
    target_policy = torch.tensor(np.array(target_policy), device=device).float()
    target_value = torch.tensor(np.array(target_value), device=device).float()

    logits, value = model(obs)

    policy_loss = -(target_policy * F.log_softmax(logits, dim=-1)).sum(dim=-1).mean()
    value_loss = F.mse_loss(value, target_value)

    loss = policy_loss + value_loss

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return {
        "loss": float(loss.item()),
        "policy_loss": float(policy_loss.item()),
        "value_loss": float(value_loss.item()),
    }



device = "cuda" if torch.cuda.is_available() else "cpu"

model = PolicyValueNet().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

replay_buffer = []

for epoch in range(100):
    for _ in range(20):
        game_data = self_play_game(
            model,
            device,
            num_simulations=64,
        )
        replay_buffer.extend(game_data)

    replay_buffer = replay_buffer[-5000:]

    logs = []
    for _ in range(50):
        batch = random.sample(replay_buffer, min(64, len(replay_buffer)))
        log = train_step(model, optimizer, batch, device)
        logs.append(log)

    avg_loss = np.mean([x["loss"] for x in logs])
    avg_policy_loss = np.mean([x["policy_loss"] for x in logs])
    avg_value_loss = np.mean([x["value_loss"] for x in logs])

    print(
        f"epoch {epoch:03d} | "
        f"loss {avg_loss:.4f} | "
        f"policy {avg_policy_loss:.4f} | "
        f"value {avg_value_loss:.4f} | "
        f"buffer {len(replay_buffer)}"
    )

    torch.save(model.state_dict(), "alphazero_tictactoe.pt")

