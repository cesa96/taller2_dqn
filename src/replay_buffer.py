import numpy as np
import torch
from src.call_logger import log_method_call

class ReplayBuffer:
    def __init__(self, capacity: int, obs_dim: int, seed: int) -> None:
        log_method_call("ReplayBuffer.__init__")
        self.capacity = int(capacity)
        self.obs_dim = obs_dim
        self.rng = np.random.default_rng(seed)

        self.states = np.zeros((self.capacity, obs_dim), dtype=np.float32)
        self.actions = np.zeros(self.capacity, dtype=np.int64)
        self.rewards = np.zeros(self.capacity, dtype=np.float32)
        self.next_states = np.zeros((self.capacity, obs_dim), dtype=np.float32)
        self.dones = np.zeros(self.capacity, dtype=np.float32)

        self.pos = 0   # dónde se escribe la próxima transición
        self.size = 0  # cuántas transiciones hay guardadas

    def push(
        self,
        observation: np.ndarray,
        action: int,
        reward: float,
        next_observation: np.ndarray,
        terminated: bool,
    ) -> None:
        self.states[self.pos] = observation
        self.actions[self.pos] = action
        self.rewards[self.pos] = reward
        self.next_states[self.pos] = next_observation
        self.dones[self.pos] = float(terminated)

        self.pos = (self.pos + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(
        self, batch_size: int, device: torch.device
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        idx = self.rng.choice(self.size, size=batch_size, replace=False)

        def to_tensor(array):
            return torch.as_tensor(array[idx], device=device)

        return (
            to_tensor(self.states),                  # (batch, obs_dim) float32
            to_tensor(self.actions).unsqueeze(1),    # (batch, 1)       int64
            to_tensor(self.rewards).unsqueeze(1),    # (batch, 1)       float32
            to_tensor(self.next_states),             # (batch, obs_dim) float32
            to_tensor(self.dones).unsqueeze(1),      # (batch, 1)       float32
        )

    def __len__(self) -> int:
        return self.size
