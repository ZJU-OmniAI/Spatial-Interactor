from __future__ import annotations

import importlib.util
import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REWARD_PATH = ROOT / "training/opd/rewards/spatial_video_reward.py"
SPEC = importlib.util.spec_from_file_location("spatial_reward", REWARD_PATH)
REWARD = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(REWARD)


class RewardTest(unittest.TestCase):
    def score(self, response: str, truth: dict) -> dict[str, float]:
        return REWARD.compute_score(
            [{"response": response, "ground_truth": json.dumps(truth)}]
        )[0]

    def test_exact_choice_and_format_reward(self):
        result = self.score(
            "Reasoning: The camera moves right.\nAnswer: B",
            {"reward_family": "choice", "choice_gold": "B"},
        )
        self.assertEqual(result["task"], 1.0)
        self.assertEqual(result["format"], 1.0)
        self.assertEqual(result["overall"], 1.0)

    def test_numeric_mra_is_continuous(self):
        result = self.score(
            "Reasoning: The displacement is close to two meters.\nAnswer: 1.5",
            {"reward_family": "numeric_mra", "value": 2.0},
        )
        self.assertAlmostEqual(result["task"], 0.75)
        self.assertAlmostEqual(result["overall"], 0.775)

    def test_multifield_reward_is_mean_accuracy(self):
        result = self.score(
            'Reasoning: Compare the endpoint and path.\nAnswer: {"straight":"A","angle":"B","path":"D"}',
            {
                "reward_family": "multi_choice",
                "fields": {"straight": "A", "angle": "B", "path": "C"},
            },
        )
        self.assertAlmostEqual(result["task"], 2.0 / 3.0)
        self.assertAlmostEqual(result["overall"], 0.7)


if __name__ == "__main__":
    unittest.main()

