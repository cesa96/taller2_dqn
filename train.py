import argparse
import csv
import random
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
import yaml

from src.agent import DQNAgent
from src.replay_buffer import ReplayBuffer

ROOT = Path(__file__).resolve().parent


def pick_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def main() -> None:
    parser = argparse.ArgumentParser(description="Entrena un agente DQN en LunarLander-v3")
    parser.add_argument("--config", type=Path, default=ROOT / "config.yaml")
    parser.add_argument("--seed", type=int, default=None, help="sobrescribe la semilla del config")
    parser.add_argument("--episodes", type=int, default=None, help="sobrescribe el número de episodios")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "results")
    parser.add_argument("--tag", type=str, default="", help="prefijo para los archivos de salida (experimentos)")
    parser.add_argument("--set", nargs="*", default=[], metavar="CLAVE=VALOR",
                        help="sobrescribe hiperparámetros de training, p. ej. --set learning_rate=0.0005 batch_size=128")
    args = parser.parse_args()

    with args.config.open(encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    env_cfg, tr = cfg["environment"], cfg["training"]
    for item in args.set:
        key, value = item.split("=", 1)
        if key not in tr:
            raise KeyError(f"Hiperparámetro desconocido: {key}")
        tr[key] = yaml.safe_load(value)

    seed = args.seed if args.seed is not None else env_cfg["seed"]
    episodes = args.episodes if args.episodes is not None else tr["episodes"]
    device = pick_device(tr["device"])

    torch.set_num_threads(1)  # red pequeña: un hilo es más rápido y permite correr semillas en paralelo
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    env = gym.make(env_cfg["id"], max_episode_steps=tr["max_steps_per_episode"])
    obs_dim = env.observation_space.shape[0]
    n_actions = env.action_space.n

    agent = DQNAgent(
        obs_dim=obs_dim,
        n_actions=n_actions,
        hidden_dims=tr["hidden_dims"],
        learning_rate=tr["learning_rate"],
        gamma=tr["gamma"],
        epsilon_start=tr["epsilon_start"],
        epsilon_end=tr["epsilon_end"],
        epsilon_decay_steps=tr["epsilon_decay_steps"],
        gradient_clip=tr["gradient_clip"],
        seed=seed,
        device=device,
    )
    buffer = ReplayBuffer(tr["replay_capacity"], obs_dim, seed)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    prefix = f"{args.tag}_" if args.tag else ""
    log_path = args.out_dir / f"{prefix}run_{seed}.csv"
    best_path = args.out_dir / f"{prefix}best_{seed}.pt"
    final_path = args.out_dir / f"{prefix}final_{seed}.pt"
    global_step = 0
    recent_rewards: list[float] = []
    best_avg = -float("inf")

    with log_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["episode", "reward", "epsilon", "loss", "steps"])

        for episode in range(1, episodes + 1):
            obs, _ = env.reset(seed=seed if episode == 1 else None)
            ep_reward, ep_steps, losses = 0.0, 0, []
            terminated = truncated = False

            while not (terminated or truncated):
                eps = agent.epsilon_at(global_step)
                action = agent.act(obs, eps)
                next_obs, reward, terminated, truncated, _ = env.step(action)

                # Solo "terminated" marca el fin real del episodio para Bellman
                buffer.push(obs, action, reward, next_obs, terminated)

                ep_reward += reward
                ep_steps += 1
                global_step += 1
                obs = next_obs

                if len(buffer) >= tr["min_replay_size"]:
                    losses.append(agent.update(buffer.sample(tr["batch_size"], device)))

                if global_step % tr["target_update_interval"] == 0:
                    agent.sync_target()

            mean_loss = float(np.mean(losses)) if losses else ""
            writer.writerow([episode, round(ep_reward, 4), round(agent.epsilon_at(global_step), 4),
                             mean_loss if mean_loss == "" else round(mean_loss, 6), ep_steps])
            f.flush()

            # Guarda el modelo con mejor media móvil de 100 episodios
            recent_rewards.append(ep_reward)
            if len(recent_rewards) >= 100:
                avg100 = float(np.mean(recent_rewards[-100:]))
                if avg100 > best_avg:
                    best_avg = avg100
                    torch.save({"state_dict": agent.q_net.state_dict(), "episode": episode,
                                "avg100": avg100, "config": tr, "seed": seed}, best_path)

            if episode % 10 == 0 or episode == 1:
                print(f"ep {episode:4d} | recompensa {ep_reward:8.2f} | pasos {ep_steps:4d} | "
                      f"eps {agent.epsilon_at(global_step):.3f} | pasos totales {global_step}")

    env.close()
    torch.save({"state_dict": agent.q_net.state_dict(), "episode": episodes,
                "config": tr, "seed": seed}, final_path)
    print(f"Logs guardados en {log_path}")
    print(f"Modelo final en {final_path}; mejor media móvil(100) = {best_avg:.2f} en {best_path}")


if __name__ == "__main__":
    main()
