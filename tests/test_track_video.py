"""Tests du module de tracking (parties pures uniquement : run_tracking
nécessite un vrai modèle RT-DETR chargé, couvert par un test d'intégration
manuel plutôt qu'unitaire)."""

import numpy as np

from src.tracking.track_video import boxes_to_mot_rows, write_mot_file


class TestBoxesToMotRows:
    def test_converts_center_xywh_to_top_left(self):
        boxes_xywh = np.array([[100.0, 100.0, 40.0, 80.0]])  # centre (100,100), 40x80
        track_ids = np.array([5])
        confs = np.array([0.9])

        rows = boxes_to_mot_rows(
            frame_idx=1, boxes_xywh=boxes_xywh, track_ids=track_ids, confs=confs
        )

        assert len(rows) == 1
        frame, tid, left, top, w, h, conf = rows[0]
        assert frame == 1
        assert tid == 5
        assert left == 80.0  # 100 - 40/2
        assert top == 60.0  # 100 - 80/2
        assert w == 40.0
        assert h == 80.0
        assert conf == 0.9

    def test_multiple_boxes_same_frame(self):
        boxes_xywh = np.array([[50.0, 50.0, 20.0, 20.0], [200.0, 150.0, 30.0, 60.0]])
        track_ids = np.array([1, 2])
        confs = np.array([0.8, 0.6])

        rows = boxes_to_mot_rows(
            frame_idx=3, boxes_xywh=boxes_xywh, track_ids=track_ids, confs=confs
        )

        assert len(rows) == 2
        assert all(row[0] == 3 for row in rows)
        assert {row[1] for row in rows} == {1, 2}

    def test_empty_boxes_returns_empty_list(self):
        rows = boxes_to_mot_rows(
            frame_idx=1,
            boxes_xywh=np.empty((0, 4)),
            track_ids=np.empty((0,), dtype=int),
            confs=np.empty((0,)),
        )
        assert rows == []

    def test_track_id_cast_to_int(self):
        # Ultralytics renvoie parfois des float pour les IDs -- on force en int
        boxes_xywh = np.array([[10.0, 10.0, 5.0, 5.0]])
        track_ids = np.array([3.0])
        confs = np.array([0.5])

        rows = boxes_to_mot_rows(
            frame_idx=1, boxes_xywh=boxes_xywh, track_ids=track_ids, confs=confs
        )

        assert isinstance(rows[0][1], int)
        assert rows[0][1] == 3


class TestWriteMotFile:
    def test_writes_correct_format(self, tmp_path):
        rows = [(1, 5, 80.0, 60.0, 40.0, 80.0, 0.9)]
        output_path = tmp_path / "output.txt"

        write_mot_file(rows, output_path)

        content = output_path.read_text()
        assert content == "1,5,80.00,60.00,40.00,80.00,0.900,-1,-1,-1\n"

    def test_writes_multiple_rows_in_order(self, tmp_path):
        rows = [
            (1, 1, 0.0, 0.0, 10.0, 10.0, 0.5),
            (1, 2, 5.0, 5.0, 10.0, 10.0, 0.6),
            (2, 1, 1.0, 1.0, 10.0, 10.0, 0.55),
        ]
        output_path = tmp_path / "output.txt"

        write_mot_file(rows, output_path)

        lines = output_path.read_text().splitlines()
        assert len(lines) == 3
        assert lines[0].startswith("1,1,")
        assert lines[1].startswith("1,2,")
        assert lines[2].startswith("2,1,")

    def test_creates_parent_directories(self, tmp_path):
        output_path = tmp_path / "nested" / "dir" / "output.txt"
        write_mot_file([(1, 1, 0.0, 0.0, 1.0, 1.0, 1.0)], output_path)
        assert output_path.exists()

    def test_empty_rows_creates_empty_file(self, tmp_path):
        output_path = tmp_path / "output.txt"
        write_mot_file([], output_path)
        assert output_path.exists()
        assert output_path.read_text() == ""
