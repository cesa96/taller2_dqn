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
        self.obs_dim = obs_dim
        self.n_actions = n_actions

        capas = []
        entrada = obs_dim
        for neuronas in hidden_dims:                 # 8 -> 128 -> 128
            capas.append(nn.Linear(entrada, neuronas))
            capas.append(nn.ReLU())
            entrada = neuronas
        capas.append(nn.Linear(entrada, n_actions))  # 128 -> 4 (sin activación)
        self.net = nn.Sequential(*capas)

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.net(observations)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


if __name__ == "__main__":
    SEED = 42
    torch.manual_seed(SEED)

    OBS_DIM = 8     # tamaño del vector de observación de LunarLander-v3
    N_ACTIONS = 4   # 0: nada, 1: motor izquierdo, 2: motor principal, 3: motor derecho

    print("PyTorch", torch.__version__)

    red = QNetwork(OBS_DIM, N_ACTIONS)
    print(red)

    print(f"{'Capa':<12}{'Pesos':>12}{'Sesgos':>10}{'Total':>10}")
    total = 0
    for nombre, capa in red.net.named_children():
        if isinstance(capa, nn.Linear):
            pesos, sesgos = capa.weight.numel(), capa.bias.numel()
            total += pesos + sesgos
            forma = f"{capa.in_features}x{capa.out_features}"
            print(f"Linear {nombre:<5}{forma:>12}{sesgos:>10}{pesos + sesgos:>10}")
    print(f"{'TOTAL':<34}{total:>10}")

    print("\ncount_parameters():", red.count_parameters())
