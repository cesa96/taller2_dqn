from collections.abc import Sequence

import torch
from torch import nn

from src.call_logger import log_method_call


class QNetwork(nn.Module):
    def __init__(
        self,
        obs_dim: int,
        n_actions: int,
        hidden_dims: Sequence[int] = (128, 128),
    ) -> None:
        super().__init__()
        log_method_call("QNetwork.__init__")

    def forward(self, observations: torch.Tensor) -> None:
        log_method_call("QNetwork.forward")
