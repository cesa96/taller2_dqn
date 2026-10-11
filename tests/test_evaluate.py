"""Pruebas del protocolo: python -m unittest discover -s tests -v.

El entorno simulado permite detectar exploración, gradientes, errores de semillas
y confusiones entre retorno acumulado y recompensa terminal sin volver a entrenar.
"""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch

import evaluate


class TwoStepEnvironment:
    metadata = {"render_fps": 50}
    action_space = SimpleNamespace(n=4)

    def __init__(self):
        self.seeds = []
        self.actions = []
        self.closed = False

    def reset(self, *, seed):
        self.seeds.append(seed)
        self.step_count = 0
        return np.zeros(8, dtype=np.float32), {}

    def step(self, action):
        self.actions.append(action)
        self.step_count += 1
        last = self.step_count == 2
        landing = len(self.seeds) == 1
        reward = 100.0 if last and landing else 1.0
        return np.zeros(8, dtype=np.float32), reward, last and landing, last and not landing, {}

    def close(self):
        self.closed = True


class FixedGreedyNetwork(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.values = torch.nn.Parameter(torch.tensor([0.0, 1.0, 4.0, 3.0]))
        self.modes = []

    def forward(self, observation):
        self.modes.append((torch.is_grad_enabled(), torch.is_inference_mode_enabled(), self.training))
        return self.values.unsqueeze(0)


class EvaluationProtocolTests(unittest.TestCase):
    def test_terminal_event_takes_precedence_over_time_limit(self):
        self.assertEqual(evaluate.classify_outcome(True, True, 100.0), "landed")
        self.assertEqual(evaluate.classify_outcome(True, True, -100.0), "crash_or_out_of_bounds")
        self.assertEqual(evaluate.classify_outcome(False, True, 100.0), "time_limit")
        with self.assertRaises(ValueError):
            evaluate.classify_outcome(False, False, 100.0)

    def test_total_reward_threshold_is_independent_of_landing(self):
        rows = [
            {"reward": 210.0, "steps": 2, "outcome": evaluate.classify_outcome(True, False, -100.0)},
            {"reward": 190.0, "steps": 4, "outcome": evaluate.classify_outcome(True, False, 100.0)},
        ]
        stats = evaluate.summarize(rows)
        self.assertEqual(stats["reward_mean"], 200.0)
        self.assertEqual(stats["reward_std"], 10.0)
        self.assertEqual(stats["reward_std_ddof"], 0)
        self.assertEqual(stats["episodes_reward_ge_200"], 1)
        self.assertEqual(stats["outcomes"]["landed"], 1)
        self.assertEqual(stats["outcomes"]["crash_or_out_of_bounds"], 1)
        self.assertEqual(stats["mean_steps"], 3.0)

    def test_greedy_loop_has_no_exploration_gradients_or_weight_updates(self):
        environment = TwoStepEnvironment()
        network = FixedGreedyNetwork().eval()
        original_weights = network.values.detach().clone()
        with patch.object(evaluate.gym, "make", return_value=environment) as make:
            rows, frames, actions, fps = evaluate.evaluate_policy(network, episodes=2, seed=101, max_steps=500)
        make.assert_called_once_with("LunarLander-v3", max_episode_steps=500, render_mode=None)
        self.assertEqual(environment.seeds, [101, 102])
        self.assertEqual(environment.actions, [2, 2, 2, 2])
        self.assertTrue(all(mode == (False, True, False) for mode in network.modes))
        self.assertTrue(torch.equal(network.values, original_weights))
        self.assertIsNone(network.values.grad)
        self.assertEqual([row["steps"] for row in rows], [2, 2])
        self.assertEqual([row["outcome"] for row in rows], ["landed", "time_limit"])
        self.assertEqual(actions["2"], 4)
        self.assertEqual(frames, [])
        self.assertEqual(fps, 50)
        self.assertTrue(environment.closed)

    def test_random_policy_reuses_reset_seeds_and_its_own_rng(self):
        environment = TwoStepEnvironment()
        with patch.object(evaluate.gym, "make", return_value=environment):
            rows, _, _, _ = evaluate.evaluate_policy(None, episodes=2, seed=101, max_steps=500)
        self.assertEqual(environment.seeds, [101, 102])
        self.assertEqual([row["environment_seed"] for row in rows], [101, 102])
        expected_actions = np.random.default_rng(101).integers(4, size=4).tolist()
        self.assertEqual(environment.actions, expected_actions)
        self.assertTrue(environment.closed)


if __name__ == "__main__":
    unittest.main()
