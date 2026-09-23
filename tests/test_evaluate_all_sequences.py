"""Tests des parties pures d'evaluate_all_sequences.py (evaluate_sequence et
main nécessitent un vrai modèle)."""

from src.tracking.evaluate_all_sequences import rows_to_pred_frames


class TestRowsToPredFrames:
    def test_groups_by_frame(self):
        rows = [
            (1, 5, 10.0, 10.0, 40.0, 80.0, 0.9),
            (1, 6, 100.0, 10.0, 40.0, 80.0, 0.8),
            (2, 5, 12.0, 11.0, 40.0, 80.0, 0.85),
        ]
        frames = rows_to_pred_frames(rows)
        assert set(frames.keys()) == {1, 2}

    def test_frame_1_has_two_ids(self):
        rows = [
            (1, 5, 10.0, 10.0, 40.0, 80.0, 0.9),
            (1, 6, 100.0, 10.0, 40.0, 80.0, 0.8),
        ]
        frames = rows_to_pred_frames(rows)
        ids, boxes = frames[1]
        assert set(ids) == {5, 6}
        assert boxes.shape == (2, 4)

    def test_empty_rows_returns_empty_dict(self):
        assert rows_to_pred_frames([]) == {}

    def test_box_values_preserved(self):
        rows = [(1, 1, 10.0, 20.0, 30.0, 40.0, 0.9)]
        frames = rows_to_pred_frames(rows)
        _, boxes = frames[1]
        assert boxes[0].tolist() == [10.0, 20.0, 30.0, 40.0]
