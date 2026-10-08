from collections.abc import Sequence

import numpy as np
import torch
from torch import nn

from src.call_logger import log_method_call
from src.network import QNetwork


class DQNAgent:
    """Agente DQN: red online + red objetivo, política ε-greedy y pérdida Huber."""

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
        self.n_actions = n_actions
        self.gamma = gamma
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay_steps = epsilon_decay_steps
        self.gradient_clip = gradient_clip
        self.device = device
        self.rng = np.random.default_rng(seed)

        torch.manual_seed(seed)
        self.q_net = QNetwork(obs_dim, n_actions, hidden_dims).to(device)
        self.target_net = QNetwork(obs_dim, n_actions, hidden_dims).to(device)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.target_net.eval()

        self.optimizer = torch.optim.Adam(self.q_net.parameters(), lr=learning_rate)
        self.loss_fn = nn.SmoothL1Loss()  # Huber

    def epsilon_at(self, step: int) -> float:
        """Decaimiento lineal de ε desde epsilon_start hasta epsilon_end."""
        frac = min(step / self.epsilon_decay_steps, 1.0)
        return self.epsilon_start + frac * (self.epsilon_end - self.epsilon_start)

    def act(self, state: np.ndarray, eps: float) -> int:
        """Acción ε-greedy: aleatoria con probabilidad eps, si no argmax Q."""
        if self.rng.random() < eps:
            return int(self.rng.integers(self.n_actions))
        with torch.no_grad():
            s = torch.as_tensor(state, dtype=torch.float32, device=self.device)
            return int(self.q_net(s.unsqueeze(0)).argmax(dim=1).item())

    def update(
        self,
        batch: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    ) -> float:
        """Un paso de descenso de gradiente sobre el lote. Devuelve la pérdida."""
        states, actions, rewards, next_states, dones = batch

        # Q(s, a) de la red online para la acción que se tomó
        q_sa = self.q_net(states).gather(1, actions)

        # y = r + γ · max_a' Q_target(s', a') · (1 - done), sin gradiente
        with torch.no_grad():
            max_next_q = self.target_net(next_states).max(dim=1, keepdim=True).values
            target = rewards + self.gamma * max_next_q * (1.0 - dones)

        loss = self.loss_fn(q_sa, target)
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.q_net.parameters(), self.gradient_clip)
        self.optimizer.step()
        return float(loss.item())

    def sync_target(self) -> None:
        """Copia los pesos de la red online a la red objetivo (actualización dura)."""
        self.target_net.load_state_dict(self.q_net.state_dict())
