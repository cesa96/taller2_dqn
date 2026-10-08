"""Analiza el comportamiento de un modelo entrenado (P5, apoyo a la reflexión).

Uso: python experiments/analisis_politica.py results/best_42.pt [--episodes 100]
     python experiments/analisis_politica.py random            # línea base aleatoria

Corre episodios con política greedy (ε = 0) y reporta recompensa media ± desv.,
% de episodios con recompensa >= 200, cómo terminan (aterrizaje, choque o
corte por tiempo) y qué tan seguido usa cada acción.
"""
import argparse
import sys
from pathlib import Path
from collections import Counter

import gymnasium as gym
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.network import QNetwork  # noqa: E402

NOMBRES = ["nada", "motor izq.", "motor principal", "motor der."]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("modelo", help="ruta a un .pt o 'random'")
    p.add_argument("--episodes", type=int, default=100)
    p.add_argument("--seed", type=int, default=1000, help="semillas distintas a las de entrenamiento")
    p.add_argument("--max-steps", type=int, default=500)
    a = p.parse_args()

    env = gym.make("LunarLander-v3", max_episode_steps=a.max_steps)
    net = None
    if a.modelo != "random":
        ckpt = torch.load(a.modelo, map_location="cpu")
        net = QNetwork(8, 4, ckpt["config"]["hidden_dims"])
        net.load_state_dict(ckpt["state_dict"])
        net.eval()
    rng = np.random.default_rng(a.seed)

    recompensas, finales, acciones, pasos = [], Counter(), Counter(), []
    for i in range(a.episodes):
        obs, _ = env.reset(seed=a.seed + i)
        total, done, n = 0.0, False, 0
        while not done:
            if net is None:
                act = int(rng.integers(4))
            else:
                with torch.no_grad():
                    act = int(net(torch.as_tensor(obs).unsqueeze(0)).argmax().item())
            obs, r, term, trunc, _ = env.step(act)
            total += r; n += 1; acciones[act] += 1
            done = term or trunc
        if trunc and not term:
            finales["corte por tiempo"] += 1
        elif r >= 50:
            finales["aterrizaje"] += 1
        else:
            finales["choque / fuera"] += 1
        recompensas.append(total); pasos.append(n)

    rec = np.array(recompensas)
    print(f"Modelo: {a.modelo}  ({a.episodes} episodios, semillas {a.seed}..{a.seed + a.episodes - 1})")
    print(f"Recompensa media: {rec.mean():.1f} ± {rec.std():.1f}  (mín {rec.min():.1f}, máx {rec.max():.1f})")
    print(f"Episodios con recompensa >= 200: {(rec >= 200).mean() * 100:.0f} %")
    print(f"Pasos por episodio: {np.mean(pasos):.0f}")
    print("Final de los episodios:", dict(finales))
    tot = sum(acciones.values())
    print("Uso de acciones:", {NOMBRES[k]: f"{v / tot * 100:.0f} %" for k, v in sorted(acciones.items())})


if __name__ == "__main__":
    main()
