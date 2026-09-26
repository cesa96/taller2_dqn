import numpy as np
import torch

from src.call_logger import log_method_call


class ReplayBuffer:
    def __init__(self, capacity: int, obs_dim: int, seed: int) -> None:
        log_method_call("ReplayBuffer.__init__")

    def push(
        self,
        observation: np.ndarray,
        action: int,
        reward: float,
        next_observation: np.ndarray,
        terminated: bool,
    ) -> None:
        log_method_call("ReplayBuffer.push")

    def sample(
        self, batch_size: int, device: torch.device
    ) -> None:
        log_method_call("ReplayBuffer.sample")
