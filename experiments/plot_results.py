"""Reproduce figures and numerical summaries from the published training logs.

Run from any working directory: python experiments/plot_results.py
No new training is performed. Empty losses remain missing, never zero.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
COLORS = {7: "#167D8D", 42: "#C06432", 123: "#7560A7"}


def load_log(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    required = {"episode", "reward", "epsilon", "loss", "steps"}
    if not required.issubset(frame.columns):
        raise ValueError(f"Missing columns in {path}: {required - set(frame.columns)}")
    if frame.empty or frame["episode"].tolist() != list(range(1, len(frame) + 1)):
        raise ValueError(f"Episodes must be consecutive and start at one: {path}")
    for field in required - {"loss"}:
        if not np.isfinite(frame[field]).all():
            raise ValueError(f"Missing or non-finite values in {path}: {field}")
    if not np.isfinite(frame["loss"].dropna()).all():
        raise ValueError(f"Non-finite recorded loss in {path}")
    if (frame["steps"] < 1).any():
        raise ValueError(f"Episode with no steps in {path}")
    frame["reward_rolling_100"] = frame["reward"].rolling(100, min_periods=100).mean()
    frame["cumulative_steps"] = frame["steps"].cumsum()
    # Require 50 valid observations: the replay warm-up remains a gap.
    frame["loss_rolling_50"] = frame["loss"].rolling(50, min_periods=50).mean()
    return frame


def first_episode(frame: pd.DataFrame, condition: pd.Series) -> int | None:
    selected = frame.loc[condition, "episode"]
    return int(selected.iloc[0]) if len(selected) else None


def summarize(seed: int, frame: pd.DataFrame, decay_steps: int) -> dict:
    rolling = frame["reward_rolling_100"]
    valid_loss = frame["loss"].dropna()
    best_row = frame.loc[frame["reward"].idxmax()]
    worst_row = frame.loc[frame["reward"].idxmin()]
    best_rolling_row = frame.loc[rolling.idxmax()] if rolling.notna().any() else None
    return {
        "seed": seed,
        "episodes": len(frame),
        "total_steps": int(frame["steps"].sum()),
        "episode_steps_min": int(frame["steps"].min()),
        "episode_steps_max": int(frame["steps"].max()),
        "episodes_at_step_cap_500": int((frame["steps"] == 500).sum()),
        "reward_mean_all": float(frame["reward"].mean()),
        "reward_std_all_ddof1": float(frame["reward"].std(ddof=1)),
        "best_episode_reward": float(best_row["reward"]),
        "best_reward_episode": int(best_row["episode"]),
        "worst_episode_reward": float(worst_row["reward"]),
        "worst_reward_episode": int(worst_row["episode"]),
        "first_100_reward_mean": float(frame["reward"].head(100).mean()),
        "first_100_reward_std_ddof1": float(frame["reward"].head(100).std(ddof=1)),
        "final_100_reward_mean": float(frame["reward"].tail(100).mean()),
        "final_100_reward_std_ddof1": float(frame["reward"].tail(100).std(ddof=1)),
        "final_100_reward_std_ddof0": float(frame["reward"].tail(100).std(ddof=0)),
        "best_100_reward_mean": float(best_rolling_row["reward_rolling_100"]) if best_rolling_row is not None else None,
        "best_100_ending_episode": int(best_rolling_row["episode"]) if best_rolling_row is not None else None,
        "first_solved_episode": first_episode(frame, rolling >= 200),
        "full_100_windows_at_least_200": int((rolling >= 200).sum()),
        "first_episode_ending_at_or_after_epsilon_decay": first_episode(frame, frame["cumulative_steps"] >= decay_steps),
        "first_logged_min_epsilon_episode": first_episode(frame, frame["epsilon"] == frame["epsilon"].min()),
        "epsilon_min": float(frame["epsilon"].min()),
        "epsilon_max": float(frame["epsilon"].max()),
        "loss_missing_episode_count": int(frame["loss"].isna().sum()),
        "first_valid_loss_episode": first_episode(frame, frame["loss"].notna()),
        "loss_first_100_valid_episode_mean": float(valid_loss.head(100).mean()),
        "loss_last_100_valid_episode_mean": float(valid_loss.tail(100).mean()),
        "loss_min": float(valid_loss.min()),
        "loss_max": float(valid_loss.max()),
    }


def style() -> None:
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 11,
        "axes.titlesize": 12, "axes.labelsize": 11,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.edgecolor": "#BCC5CC", "axes.labelcolor": "#253341",
        "text.color": "#253341", "xtick.color": "#52616D", "ytick.color": "#52616D",
        "figure.facecolor": "white", "axes.facecolor": "#FAFBFC",
        "savefig.facecolor": "white", "grid.color": "#DDE3E7", "grid.alpha": 0.65,
    })


def write_figures(logs: dict[int, pd.DataFrame], summaries: list[dict], dest: Path) -> None:
    style()
    count = len(logs)
    fig, axes = plt.subplots(count, 1, figsize=(12.5, 3.0 * count + 1.1), sharex=True, sharey=True, squeeze=False)
    fig.suptitle("LunarLander-v3 · Recompensa durante el entrenamiento", x=0.085, ha="left", y=0.99, fontsize=19, fontweight="bold")
    fig.text(0.085, 0.954, "Tres semillas · 1.000 episodios por semilla · media móvil de 100 episodios completos", fontsize=11)
    for ax, (seed, frame), summary in zip(axes[:, 0], logs.items(), summaries):
        color = COLORS.get(seed, "#167D8D")
        ax.plot(frame["episode"], frame["reward"], color=color, alpha=0.28, lw=0.75, label="Recompensa por episodio")
        ax.plot(frame["episode"], frame["reward_rolling_100"], color=color, lw=2.5, label="Media móvil (100)")
        ax.axhline(200, color="#465564", ls="--", lw=1.2, label="Umbral: 200")
        ax.set_title(f"Semilla {seed}", loc="left", fontweight="bold", pad=8)
        ax.set_ylabel("Recompensa")
        ax.grid(axis="y")
        ax.set_xlim(1, int(frame["episode"].max()))
        solved = summary["first_solved_episode"]
        label = f"Primera media ≥ 200: episodio {solved}" if solved is not None else "No alcanza una media móvil de 200"
        ax.text(0.98, 0.055, label, transform=ax.transAxes, ha="right", va="bottom", fontsize=10,
                bbox={"facecolor": "white", "alpha": 0.90, "edgecolor": "none", "pad": 4})
        if solved is not None:
            value = frame.loc[frame["episode"] == solved, "reward_rolling_100"].iloc[0]
            ax.scatter([solved], [value], color=color, s=45, edgecolor="white", zorder=5)
    axes[0, 0].legend(loc="lower left", frameon=True, facecolor="white", framealpha=0.9, fontsize=9, ncol=3)
    axes[-1, 0].set_xlabel("Episodio de entrenamiento")
    fig.subplots_adjust(left=0.085, right=0.98, top=0.90, bottom=0.085, hspace=0.30)
    fig.text(0.085, 0.018, "Fuente: experiments/logs/final/run_{7,42,123}.csv. ε-greedy durante el entrenamiento; no es evaluación con ε = 0.", fontsize=9)
    fig.savefig(dest / "rewards_multiple_seeds.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(count, 1, figsize=(12.5, 2.65 * count + 1.0), sharex=True, squeeze=False)
    fig.suptitle("LunarLander-v3 · Pérdida Huber durante el entrenamiento", x=0.085, ha="left", y=0.99, fontsize=19, fontweight="bold")
    fig.text(0.085, 0.948, "Promedio de las actualizaciones de cada episodio; la media móvil requiere 50 valores válidos", fontsize=11)
    for ax, (seed, frame) in zip(axes[:, 0], logs.items()):
        color = COLORS.get(seed, "#167D8D")
        ax.plot(frame["episode"], frame["loss"], color=color, alpha=0.35, lw=0.8, label="Pérdida por episodio")
        ax.plot(frame["episode"], frame["loss_rolling_50"], color=color, lw=2.3, label="Media móvil (50)")
        ax.set_title(f"Semilla {seed}", loc="left", fontweight="bold", pad=8)
        ax.set_ylabel("Loss Huber")
        ax.set_ylim(bottom=0)
        ax.grid(axis="y")
        ax.set_xlim(1, int(frame["episode"].max()))
    axes[0, 0].legend(loc="upper right", frameon=False, fontsize=10)
    axes[-1, 0].set_xlabel("Episodio de entrenamiento")
    fig.subplots_adjust(left=0.085, right=0.98, top=0.89, bottom=0.095, hspace=0.34)
    fig.text(0.085, 0.021, "Los episodios sin actualizaciones se conservan como datos faltantes. Cada panel usa su propia escala vertical.", fontsize=9)
    fig.savefig(dest / "loss_multiple_seeds.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(12.5, 5.7))
    fig.suptitle("LunarLander-v3 · Exploración ε-greedy", x=0.085, ha="left", y=0.98, fontsize=19, fontweight="bold")
    fig.text(0.085, 0.913, "ε registrado al terminar cada episodio · decaimiento lineal durante 50.000 pasos", fontsize=11)
    for (seed, frame), summary in zip(logs.items(), summaries):
        floor_episode = summary["first_episode_ending_at_or_after_epsilon_decay"]
        label = f"Semilla {seed}: ε mínimo desde el episodio {floor_episode}"
        ax.plot(frame["episode"], frame["epsilon"], color=COLORS.get(seed), lw=2.5, label=label)
    ax.axhline(0.05, color="#465564", lw=1.1, ls="--", label="Mínimo configurado: ε = 0,05")
    ax.set_xlim(1, max(len(frame) for frame in logs.values()))
    ax.set_ylim(0, 1.04)
    ax.set_xlabel("Episodio de entrenamiento")
    ax.set_ylabel("Probabilidad de acción aleatoria (ε)")
    ax.grid(axis="y")
    ax.legend(loc="upper right", frameon=False, fontsize=10)
    fig.subplots_adjust(left=0.085, right=0.98, top=0.86, bottom=0.17)
    fig.text(0.085, 0.035, "El mínimo puede alcanzarse dentro del episodio indicado. La evaluación usa ε = 0, sin exploración.", fontsize=9)
    fig.savefig(dest / "epsilon_multiple_seeds.png", dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--logs-dir", type=Path, default=ROOT / "experiments" / "logs" / "final")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "results")
    parser.add_argument("--seeds", type=int, nargs="+", default=[7, 42, 123])
    parser.add_argument("--epsilon-decay-steps", type=int, default=50000)
    args = parser.parse_args()
    logs = {seed: load_log(args.logs_dir / f"run_{seed}.csv") for seed in args.seeds}
    summaries = [summarize(seed, frame, args.epsilon_decay_steps) for seed, frame in logs.items()]
    figure_dir = args.out_dir / "figures"
    training_dir = args.out_dir / "training"
    figure_dir.mkdir(parents=True, exist_ok=True)
    training_dir.mkdir(parents=True, exist_ok=True)
    provenance = []
    for seed in logs:
        path = args.logs_dir / f"run_{seed}.csv"
        provenance.append({"file": path.name, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
        logs[seed].to_csv(training_dir / f"derived_run_{seed}.csv", index=False, na_rep="")
    result = {
        "method": {
            "source": "Published historical logs in experiments/logs/final; no retraining",
            "reward_precision": "Source CSV rewards are already rounded to 4 decimal places",
            "rolling_window": 100,
            "minimum_rolling_observations": 100,
            "solved_definition": "First episode ending a complete trailing 100-episode reward mean >= 200",
            "standard_deviation": "Sample standard deviation, ddof=1, except fields explicitly ending in ddof0",
            "epsilon_recording": "End of episode; decay floor onset derived from cumulative steps >= epsilon_decay_steps",
            "epsilon_decay_steps": args.epsilon_decay_steps,
            "loss_recording": "Episode mean of update losses; missing episodes remain missing and are excluded from loss summaries",
            "loss_smoothing": "Trailing 50 episodes with all 50 losses present",
        },
        "source_files": provenance,
        "runs": summaries,
        "aggregate": {
            "number_of_seeds": len(summaries),
            "solved_seed_count": sum(row["first_solved_episode"] is not None for row in summaries),
            "total_steps": sum(row["total_steps"] for row in summaries),
            "best_training_seed": max(summaries, key=lambda row: row["best_100_reward_mean"])["seed"],
        },
    }
    (training_dir / "training_summary.json").write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    pd.DataFrame(summaries).to_csv(training_dir / "training_summary.csv", index=False, na_rep="")
    write_figures(logs, summaries, figure_dir)
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
    print(f"Figures: {figure_dir}")
    print(f"Numerical summaries: {training_dir}")


if __name__ == "__main__":
    main()
