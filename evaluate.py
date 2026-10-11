"""Evaluación reproducible de DQN, sin entrenamiento ni exploración (epsilon = 0).

Desde la raíz del repositorio:
    python evaluate.py
    python evaluate.py --model models/best_seed7.pt --episodes 100 --seed 20000 --gif
    python evaluate.py --out-dir results/evaluation --gif results/evaluation/agent.gif

Las semillas por defecto 20000..20099 están separadas de 1000..1099, utilizadas
anteriormente para comparar modelos. La política aleatoria usa los mismos estados
iniciales y un generador local de acciones sembrado con --seed. El límite temporal
se obtiene del checkpoint. La desviación reportada es poblacional (ddof=0).

Se guardan dos CSV por episodio y evaluation_summary.json con métricas y
procedencia. --gif [RUTA] registra el primer episodio, fijado antes de evaluar;
--gif-seed permite elegir previamente otra semilla del conjunto de evaluación.
El GIF no constituye evidencia de que todos los episodios sean exitosos.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import math
import platform
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
import torch

from src.network import QNetwork

ROOT = Path(__file__).resolve().parent
ENVIRONMENT = "LunarLander-v3"
CSV_FIELDS = (
    "episode", "environment_seed", "reward", "steps", "terminated", "truncated",
    "terminal_reward", "outcome",
)


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("El valor debe ser un entero positivo.")
    return parsed


def nonnegative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("La semilla debe ser un entero no negativo.")
    return parsed


def classify_outcome(terminated: bool, truncated: bool, terminal_reward: float) -> str:
    """LunarLander sustituye la recompensa terminal por +100 al aterrizar.

    Una recompensa acumulada >=200 es un umbral de rendimiento, no una prueba
    de aterrizaje. Si ambos indicadores son verdaderos, prevalece el terminal.
    """
    if terminated:
        return "landed" if math.isclose(terminal_reward, 100.0, abs_tol=1e-6) else "crash_or_out_of_bounds"
    if truncated:
        return "time_limit"
    raise ValueError("El episodio aún no ha terminado.")


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    rewards = np.asarray([row["reward"] for row in rows], dtype=np.float64)
    if rewards.size == 0 or not np.all(np.isfinite(rewards)):
        raise ValueError("Se necesitan recompensas finitas de al menos un episodio.")
    outcomes = Counter(row["outcome"] for row in rows)
    return {
        "episodes": len(rows),
        "reward_mean": float(rewards.mean()),
        "reward_std": float(rewards.std(ddof=0)),
        "reward_std_ddof": 0,
        "reward_min": float(rewards.min()),
        "reward_max": float(rewards.max()),
        "reward_median": float(np.median(rewards)),
        "episodes_reward_ge_200": int(np.count_nonzero(rewards >= 200.0)),
        "fraction_reward_ge_200": float(np.mean(rewards >= 200.0)),
        "mean_steps": float(np.mean([row["steps"] for row in rows])),
        "outcomes": {key: outcomes[key] for key in ("landed", "crash_or_out_of_bounds", "time_limit")},
    }


def evaluate_policy(
    net: QNetwork | None,
    *,
    episodes: int,
    seed: int,
    max_steps: int,
    gif_seed: int | None = None,
    frame_stride: int = 2,
) -> tuple[list[dict[str, Any]], list[np.ndarray], dict[str, int], int]:
    """Evalúa greedy si hay red, o acciones uniformes con un RNG local.

    Capturar RGB no cambia la selección de acciones ni agrega otro episodio.
    """
    env = gym.make(
        ENVIRONMENT,
        max_episode_steps=max_steps,
        render_mode="rgb_array" if gif_seed is not None else None,
    )
    rng = np.random.default_rng(seed)
    rows: list[dict[str, Any]] = []
    frames: list[np.ndarray] = []
    actions: Counter[int] = Counter()
    fps = int(env.metadata.get("render_fps", 50))
    try:
        with torch.inference_mode():
            for index in range(episodes):
                environment_seed = seed + index
                obs, _ = env.reset(seed=environment_seed)
                record = environment_seed == gif_seed
                if record:
                    frames.append(env.render())
                reward_sum = 0.0
                terminated = truncated = False
                step = 0
                terminal_reward = 0.0
                while not (terminated or truncated):
                    if net is None:
                        action = int(rng.integers(env.action_space.n))
                    else:
                        inputs = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
                        action = int(net(inputs).argmax(dim=1).item())
                    obs, reward, terminated, truncated, _ = env.step(action)
                    reward_sum += float(reward)
                    terminal_reward = float(reward)
                    step += 1
                    actions[action] += 1
                    if record and (step % frame_stride == 0 or terminated or truncated):
                        frames.append(env.render())
                rows.append({
                    "episode": index + 1,
                    "environment_seed": environment_seed,
                    "reward": reward_sum,
                    "steps": step,
                    "terminated": bool(terminated),
                    "truncated": bool(truncated),
                    "terminal_reward": terminal_reward,
                    "outcome": classify_outcome(bool(terminated), bool(truncated), terminal_reward),
                })
    finally:
        env.close()
    return rows, frames, {str(action): actions[action] for action in range(4)}, fps


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def package_versions() -> dict[str, str]:
    versions = {"python": platform.python_version()}
    for name in ("gymnasium", "numpy", "torch", "Box2D", "pygame", "imageio", "Pillow"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = "not_installed"
    return versions


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", type=Path, default=ROOT / "models" / "best_seed7.pt")
    parser.add_argument("--episodes", type=positive_int, default=100)
    parser.add_argument("--seed", type=nonnegative_int, default=20000, help="Primera semilla de evaluación (default: 20000).")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "results" / "evaluation")
    parser.add_argument("--gif", nargs="?", const="", default=None, metavar="RUTA", help="Genera GIF; sin ruta usa OUT_DIR/agent.gif.")
    parser.add_argument("--gif-seed", type=nonnegative_int, default=None, help="Semilla predeterminada del GIF; debe estar entre las evaluadas.")
    parser.add_argument("--frame-stride", type=positive_int, default=2, help="Guarda un fotograma cada N pasos (default: 2).")
    args = parser.parse_args()
    if args.gif_seed is not None and args.gif is None:
        parser.error("--gif-seed requiere --gif.")
    if args.gif_seed is not None and not args.seed <= args.gif_seed < args.seed + args.episodes:
        parser.error("--gif-seed debe pertenecer al conjunto de semillas de evaluación.")
    if not args.model.is_file():
        parser.error(f"No existe el checkpoint: {args.model}")
    return args


def main() -> None:
    args = parse_args()
    torch.set_num_threads(1)
    torch.manual_seed(args.seed)
    torch.use_deterministic_algorithms(True)
    # No se permite deserialización arbitraria de objetos Python del checkpoint.
    checkpoint = torch.load(args.model, map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, dict) or not {"config", "state_dict"}.issubset(checkpoint):
        raise ValueError("El checkpoint debe contener config y state_dict.")
    config = checkpoint["config"]
    max_steps = int(config["max_steps_per_episode"])
    if max_steps <= 0:
        raise ValueError("max_steps_per_episode debe ser positivo.")
    net = QNetwork(8, 4, hidden_dims=config["hidden_dims"])
    net.load_state_dict(checkpoint["state_dict"], strict=True)
    net.eval()
    gif_seed = (args.gif_seed if args.gif_seed is not None else args.seed) if args.gif is not None else None

    greedy_rows, frames, greedy_actions, fps = evaluate_policy(
        net, episodes=args.episodes, seed=args.seed, max_steps=max_steps,
        gif_seed=gif_seed, frame_stride=args.frame_stride,
    )
    random_rows, _, random_actions, _ = evaluate_policy(
        None, episodes=args.episodes, seed=args.seed, max_steps=max_steps,
    )
    greedy = summarize(greedy_rows)
    baseline = summarize(random_rows)
    greedy["action_counts"] = greedy_actions
    baseline["action_counts"] = random_actions
    differences = np.asarray([a["reward"] - b["reward"] for a, b in zip(greedy_rows, random_rows)])
    args.out_dir.mkdir(parents=True, exist_ok=True)
    greedy_csv = args.out_dir / "evaluation_episodes.csv"
    baseline_csv = args.out_dir / "random_baseline_episodes.csv"
    write_csv(greedy_csv, greedy_rows)
    write_csv(baseline_csv, random_rows)

    report: dict[str, Any] = {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "command": [sys.executable, *sys.argv],
        "environment": {"id": ENVIRONMENT, "max_episode_steps": max_steps},
        "protocol": {
            "episodes_per_policy": args.episodes,
            "first_environment_seed": args.seed,
            "last_environment_seed": args.seed + args.episodes - 1,
            "epsilon": 0.0,
            "policy": "argmax Q, first index breaks ties",
            "baseline": "uniform independent actions from numpy.default_rng(seed)",
            "baseline_rng_seed": args.seed,
            "paired_reset_seeds": True,
            "device": "cpu",
            "torch_num_threads": torch.get_num_threads(),
            "torch_deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
            "gradient_tracking": False,
            "network_training_mode": net.training,
            "landing_definition": "terminated and terminal reward +100, independent of total episode reward",
            "reward_threshold": 200.0,
            "std_ddof": 0,
        },
        "checkpoint": {
            "filename": args.model.name,
            "sha256": hashlib.sha256(args.model.read_bytes()).hexdigest(),
            "training_seed": checkpoint.get("seed"),
            "episode": checkpoint.get("episode"),
            "avg100": checkpoint.get("avg100"),
            "config": config,
        },
        "versions": package_versions(),
        "platform": platform.platform(),
        "evaluation": greedy,
        "random_baseline": baseline,
        "comparison": {
            "mean_reward_difference": float(differences.mean()),
            "paired_difference_std": float(differences.std(ddof=0)),
            "episodes_greedy_better_than_random": int(np.count_nonzero(differences > 0)),
            "note": "Diferencia absoluta; no se usan cocientes con recompensas negativas.",
        },
        "artifacts": {
            "evaluation_episodes": greedy_csv.name,
            "random_baseline_episodes": baseline_csv.name,
        },
        "gif": None,
    }
    if args.gif is not None:
        import imageio.v2 as imageio

        gif_path = Path(args.gif) if args.gif else args.out_dir / "agent.gif"
        gif_path.parent.mkdir(parents=True, exist_ok=True)
        # Pillow expresa la duración del GIF en milisegundos, no en segundos.
        imageio.mimsave(gif_path, frames, format="GIF", duration=1000.0 * args.frame_stride / fps, loop=0)
        gif_row = next(row for row in greedy_rows if row["environment_seed"] == gif_seed)
        report["gif"] = {
            "path": str(gif_path),
            "selection": "preselected evaluation seed; no best-episode search",
            "environment_seed": gif_seed,
            "episode": gif_row,
            "frames": len(frames),
            "frame_stride": args.frame_stride,
            "environment_fps": fps,
            "target_playback_fps": fps / args.frame_stride,
            "sha256": hashlib.sha256(gif_path.read_bytes()).hexdigest(),
        }
    summary_path = args.out_dir / "evaluation_summary.json"
    summary_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    for label, stats in (("DQN (epsilon=0)", greedy), ("Aleatorio", baseline)):
        print(f"{label}: {stats['reward_mean']:.2f} +/- {stats['reward_std']:.2f}; "
              f"mejor={stats['reward_max']:.2f}; >=200={stats['episodes_reward_ge_200']}/{stats['episodes']}; "
              f"finales={stats['outcomes']}")
    print(f"Diferencia media DQN - aleatorio: {differences.mean():.2f}")
    print(f"Resultados: {summary_path}")


if __name__ == "__main__":
    main()
