from collections.abc import Sequence

import numpy as np
import torch

from src.call_logger import log_method_call


class DQNAgent:
    def __init__(
        self,
        obs_dim: int,
        n_actions: int,
        hidden_dims: Sequence[int],
        learning_rate: float,
        gamma: float,
        epsilon_start: float,
        epsilon_end: float,
        epsilon_decay_steps: int,
        gradient_clip: float,
        seed: int,
        device: torch.device,
    ) -> None:
        log_method_call("DQNAgent.__init__")

    def act(self, state: np.ndarray, eps: float) -> None:
        log_method_call("DQNAgent.act")

    def update(
        self,
        batch: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    ) -> None:
        log_method_call("DQNAgent.update")
