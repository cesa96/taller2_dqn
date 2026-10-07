# Taller 2 - DQN en LunarLander-v3
# Replay buffer

import numpy as np
import torch


class ReplayBuffer:
    def __init__(self, capacity: int, obs_dim: int, device="cpu", seed=None):
        self.capacity = int(capacity)
        self.obs_dim = obs_dim
        self.device = torch.device(device)
        self.rng = np.random.default_rng(seed)

        self.states = np.zeros((self.capacity, obs_dim), dtype=np.float32)
        self.actions = np.zeros(self.capacity, dtype=np.int64)
        self.rewards = np.zeros(self.capacity, dtype=np.float32)
        self.next_states = np.zeros((self.capacity, obs_dim), dtype=np.float32)
        self.dones = np.zeros(self.capacity, dtype=np.float32)

        self.pos = 0   # dónde se escribe la próxima transición
        self.size = 0  # cuántas transiciones hay guardadas

    def push(self, state, action, reward, next_state, done) -> None:
        self.states[self.pos] = state
        self.actions[self.pos] = action
        self.rewards[self.pos] = reward
        self.next_states[self.pos] = next_state
        self.dones[self.pos] = float(done)

        self.pos = (self.pos + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int):
        idx = self.rng.choice(self.size, size=batch_size, replace=False)

        def to_tensor(array):
            return torch.as_tensor(array[idx], device=self.device)

        return (
            to_tensor(self.states),                  # (batch, obs_dim) float32
            to_tensor(self.actions).unsqueeze(1),    # (batch, 1)       int64
            to_tensor(self.rewards).unsqueeze(1),    # (batch, 1)       float32
            to_tensor(self.next_states),             # (batch, obs_dim) float32
            to_tensor(self.dones).unsqueeze(1),      # (batch, 1)       float32
        )

    def __len__(self) -> int:
        return self.size
