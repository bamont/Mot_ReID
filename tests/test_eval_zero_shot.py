"""Tests du module d'evaluation zero-shot
"""

import pytest

from src.detection.eval_zero_shot import aggregate


class TestAggregate:
    def test_computes_mean_and_std(self):
        metrics = [
            {"mAP50-95": 0.2},
            {"mAP50-95": 0.4},
            {"mAP50-95": 0.6},
        ]
        mean_v, std_v = aggregate(metrics)
        assert mean_v == pytest.approx(0.4)

    def test_zero_std_when_all_equal(self):
        metrics = [{"mAP50-95": 0.5}, {"mAP50-95": 0.5}, {"mAP50-95": 0.5}]
        mean_v, std_v = aggregate(metrics)
        assert mean_v == 0.5
        assert std_v == 0.0

    def test_single_value(self):
        mean_v, std_v = aggregate([{"mAP50-95": 0.33}])
        assert mean_v == 0.33
        assert std_v == 0.0
