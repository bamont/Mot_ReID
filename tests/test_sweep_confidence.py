"""Tests des parties pures de sweep_confidence.py (sweep() necessite un vrai
modele charge et couvert par un test d'integration manuel)."""

from src.tracking.sweep_confidence import print_best


class TestPrintBest:
    def test_selects_highest_mota(self, capsys):
        results = [
            {"conf": 0.25, "mota": 0.12, "idf1": 0.46},
            {"conf": 0.5, "mota": 0.40, "idf1": 0.51},
            {"conf": 0.6, "mota": 0.19, "idf1": 0.30},
        ]
        print_best(results)
        captured = capsys.readouterr()
        assert "conf=0.50" in captured.out
        assert "40.0%" in captured.out

    def test_single_result(self, capsys):
        results = [{"conf": 0.3, "mota": 0.2, "idf1": 0.3}]
        print_best(results)
        captured = capsys.readouterr()
        assert "conf=0.30" in captured.out
