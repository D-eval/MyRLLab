# alphazero_gomoku.py
import math
import random
import numpy as np
import gymnasium as gym
from gymnasium import spaces

import torch
import torch.nn as nn
import torch.nn.functional as F

import matplotlib.pyplot as plt

def plot_board(board, save_path):

    H, W = board.shape

    plt.figure(figsize=(8,8))

    # 棋盘线
    for i in range(H):
        plt.plot([0, W-1], [i, i], 'k')

    for j in range(W):
        plt.plot([j, j], [0, H-1], 'k')

    # 棋子
    for r in range(H):
        for c in range(W):

            if board[r,c] == 1:
                plt.scatter(
                    c,
                    H-1-r,
                    s=300,
                    c='black'
                )

            elif board[r,c] == -1:
                plt.scatter(
                    c,
                    H-1-r,
                    s=300,
                    facecolors='white',
                    edgecolors='black'
                )

    plt.xlim(-1, W)
    plt.ylim(-1, H)

    plt.gca().set_aspect('equal')

    plt.savefig(save_path)

    plt.close()


def check_five(
    board,
    r,
    c,
    player
):
    H, W = board.shape

    dirs = [
        (1,0),
        (0,1),
        (1,1),
        (1,-1)
    ]

    for dr, dc in dirs:

        cnt = 1

        # 正方向
        rr = r + dr
        cc = c + dc

        while (
            0 <= rr < H and
            0 <= cc < W and
            board[rr,cc] == player
        ):
            cnt += 1
            rr += dr
            cc += dc

        # 反方向
        rr = r - dr
        cc = c - dc

        while (
            0 <= rr < H and
            0 <= cc < W and
            board[rr,cc] == player
        ):
            cnt += 1
            rr -= dr
            cc -= dc

        if cnt >= 5:
            return True

    return False

# =========================
# Gomoko 环境：五子棋
# =========================
class GomokuEnv(gym.Env):

    def __init__(self, board_size=15):
        super().__init__()

        self.board_size = board_size

        self.board = np.zeros(
            (board_size, board_size),
            dtype=np.int8
        )

        self.current_player = 1

        self.action_space = spaces.Discrete(
            board_size * board_size
        )

        self.observation_space = spaces.Box(
            low=-1,
            high=1,
            shape=(3, board_size, board_size),
            dtype=np.float32
        )
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.board[:] = 0
        self.current_player = 1 # { +1, -1 }
        self._winner = 0
        return self._obs(), {}

    def _obs(self):
        # 永远从当前玩家视角看棋盘
        return self.board * self.current_player # (H, W)

    def legal_actions(self):
        return [i for i in range(self.board_size * self.board_size) if self.board[i // self.board_size, i % self.board_size] == 0]

    def step(self, action):
        r, c = divmod(action, self.board_size)

        if self.board[r, c] != 0:
            # 非法落子，直接输
            return self._obs(), -1.0, True, False, {}

        self.board[r, c] = self.current_player

        i_win = check_five(self.board, r, c, self.current_player)
        if i_win != 0:
            self._winner = self.current_player
            return self._obs(), 1.0, True, False, {}

        if len(self.legal_actions()) == 0:
            return self._obs(), 0.0, True, False, {}

        self.current_player *= -1
        return self._obs(), 0.0, False, False, {}

    def clone(self):
        env = GomokuEnv()
        env.board = self.board.copy()
        env.current_player = self.current_player
        return env

# =========================
# Policy-Value 网络
# =========================

class PolicyValueNet(nn.Module):
    def __init__(self, board_size=15):
        super().__init__()
        self.board_size = board_size
        self.backbone = nn.Sequential(
            nn.Conv2d(2, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(),
        )
        # policy head
        self.policy_conv = nn.Conv2d(
            64,
            2,
            1
        )
        self.policy_fc = nn.Linear(
            2 * board_size * board_size,
            board_size * board_size
        )
        # value head
        self.value_conv = nn.Conv2d(
            64,
            1,
            1
        )
        self.value_fc1 = nn.Linear(
            board_size * board_size,
            64
        )
        self.value_fc2 = nn.Linear(
            64,
            1
        )
    def forward(self, x):
        """
            x: (B, H, W)
            return:
            p: (B, H * W)
            v: (B,)
        """
        x = x.unsqueeze(1) # (B, 1, H, W)
        x_opp = x==-1
        x_self = x==1
        
        x = torch.cat([x_self, x_opp], dim=1).float() # (B, 3, H, W)
        
        h = self.backbone(x) # (B, D, W, H)
        # ======================
        # policy
        # ======================
        p = self.policy_conv(h) # (B, 2, W, H)
        p = p.flatten(1) # (B, 2 * W * H)
        p = self.policy_fc(p) # (B, W * H)
        # ======================
        # value
        # ======================
        v = self.value_conv(h) # (B, 1, W, H)
        v = v.flatten(1) # (B, W * H)
        v = F.relu(
            self.value_fc1(v) # (B, 64)
        )
        v = torch.tanh(
            self.value_fc2(v) # (B, 1)
        ).squeeze(-1) # (B,)
        return p, v

# =========================
# MCTS 蒙特卡洛树搜索
# =========================

class Node:
    def __init__(self, prior=0.0):
        self.prior = prior
        self.visit_count = 0
        self.value_sum = 0.0
        self.children = {} # {action: Node}

    @property
    def value(self):
        if self.visit_count == 0:
            return 0.0
        return self.value_sum / self.visit_count


def softmax_with_temperature(logits, T=1.0):
    logits = logits / T
    logits = logits - np.max(logits)

    probs = np.exp(logits)
    probs = probs / probs.sum()

    return probs

def evaluate(env, model, device):
    """
        只在这里进行 forward
        return:
        prob : (H * W)
        value : float
    """
    
    obs = torch.tensor(env._obs(), device=device).unsqueeze(0) # (1, H, W)
    with torch.no_grad():
        logits, value = model(obs) # (B, H * W), (B,)

    logits = logits[0].cpu().numpy() # (W * H)
    legal = env.legal_actions() # (M), legal idx, 0 ~ W*H-1

    mask = np.full(logits.shape[0], -1e9)
    mask[legal] = 0
    logits = logits + mask

    probs = softmax_with_temperature(logits)

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
    """
    return: (H, W): prob
    """
    
    root = Node()

    probs, _ = evaluate(env, model, device) # prior prob
    for a in env.legal_actions():
        root.children[a] = Node(prior=probs[a])
    # 每个 simulation 进行1次 forward
    for _ in range(num_simulations):
        sim_env = env.clone()
        node = root
        path = [node]

        done = False
        reward = 0.0

        while node.children: # 一开始只有 root 是 True，后来就越来越
            action, node = select_child(node) # 根据 value 和 visit_count 共同决定
            _, reward, terminated, truncated, _ = sim_env.step(action)
            done = terminated or truncated
            path.append(node)

            if done:
                break

        # 如果在num_simulation中没有done，那获得的策略网络生成的value就没有参考意义啊
        # 我知道了，因为self_play是要玩到最后的，在self_play的后面几轮里，num_simulation可以很快到达done，这时候mcts里的value, select_child是有参考意义的
        if done: # 如果 done 了，reward 由游戏规则提供
            value = reward
        else: # 否则，reward 由 策略价值网络 提供
            probs, value = evaluate(sim_env, model, device)
            for a in sim_env.legal_actions():
                node.children[a] = Node(prior=probs[a])

        # 反向传播，注意玩家视角交替，所以 value 要取负
        for n in reversed(path):
            n.visit_count += 1
            n.value_sum += value
            value = -value

    visits = np.zeros(env.board_size**2, dtype=np.float32)
    for a, child in root.children.items():
        visits[a] = child.visit_count

    policy = visits / visits.sum()
    return policy


# =========================
# 自博弈
# =========================

def self_play_game(model, device, num_simulations=64):
    """
        return: List[
            (H, W),
            (H * W), prob
            float, real value
        ]
    """
    
    env = GomokuEnv()
    obs, _ = env.reset() # (H, W)

    data = []
    players = []

    done = False

    while not done:
        policy = run_mcts(env, model, device, num_simulations)

        data.append((env._obs().copy(), policy.copy())) # (H, W), (H*W)
        players.append(env.current_player)

        action = np.random.choice(env.board_size**2, p=policy)
        _, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

    # 保存 board
    plot_board(env.board, "./temp.pdf")

    winner = env._winner

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

    obs = torch.tensor(np.array(obs), device=device).float() # (L, H, W) int
    target_policy = torch.tensor(np.array(target_policy), device=device).float() # (L, H * W)
    target_value = torch.tensor(np.array(target_value), device=device).float() # (L,)

    logits, value = model(obs) # (L, H * W), (L,)

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
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4) # weight_decay: l2 regularization

replay_buffer = []

for epoch in range(100):
    for _ in range(5):
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

    torch.save(model.state_dict(), "alphazero_gomoku.pt")

