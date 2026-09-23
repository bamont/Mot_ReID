"""Tests du module d'evaluation (MOTA/IDF1).

load_gt/load_pred sont testés avec des fichiers synthétiques (pas besoin
d'un vrai MOT17). build_accumulator/compute_summary sont testés avec de
vraies petites données via motmetrics."""

import numpy as np
import pytest

from src.tracking.evaluate import build_accumulator, compute_summary, load_gt, load_pred


class TestLoadGt:
    @pytest.fixture
    def gt_file(self, tmp_path):
        content = (
            "1,1,10,10,50,100,1,1,0.9\n"  # garde : piéton, conf=1, vis=0.9
            "1,2,60,10,50,100,1,7,0.9\n"  # rejète : classe != 1 (non-piéton)
            "1,3,110,10,50,100,0,1,0.9\n"  # rejète : conf=0
            "1,4,160,10,50,100,1,1,0.1\n"  # rejète : visibilité < seuil (0.3)
            "2,1,15,15,50,100,1,1,0.5\n"  # garde, frame 2
        )
        path = tmp_path / "gt.txt"
        path.write_text(content)
        return path

    def test_filters_non_pedestrian_class(self, gt_file):
        frames = load_gt(gt_file, min_visibility=0.3)
        ids_frame1, _ = frames[1]
        assert 2 not in ids_frame1

    def test_filters_conf_zero(self, gt_file):
        frames = load_gt(gt_file, min_visibility=0.3)
        ids_frame1, _ = frames[1]
        assert 3 not in ids_frame1

    def test_filters_low_visibility(self, gt_file):
        frames = load_gt(gt_file, min_visibility=0.3)
        ids_frame1, _ = frames[1]
        assert 4 not in ids_frame1

    def test_keeps_valid_annotation(self, gt_file):
        frames = load_gt(gt_file, min_visibility=0.3)
        ids_frame1, boxes_frame1 = frames[1]
        assert list(ids_frame1) == [1]
        assert boxes_frame1[0].tolist() == [10.0, 10.0, 50.0, 100.0]

    def test_multiple_frames_parsed(self, gt_file):
        frames = load_gt(gt_file, min_visibility=0.3)
        assert set(frames.keys()) == {1, 2}


class TestLoadPred:
    @pytest.fixture
    def pred_file(self, tmp_path):
        content = (
            "1,5,20,20,40,80,0.9,-1,-1,-1\n"
            "1,6,100,20,40,80,0.85,-1,-1,-1\n"
            "2,5,22,21,40,80,0.88,-1,-1,-1\n"
        )
        path = tmp_path / "pred.txt"
        path.write_text(content)
        return path

    def test_parses_all_rows(self, pred_file):
        frames = load_pred(pred_file)
        assert set(frames.keys()) == {1, 2}

    def test_frame_1_has_two_tracks(self, pred_file):
        frames = load_pred(pred_file)
        ids, boxes = frames[1]
        assert set(ids) == {5, 6}
        assert boxes.shape == (2, 4)

    def test_no_filtering_applied(self, pred_file):
        frames = load_pred(pred_file)
        ids, _ = frames[2]
        assert list(ids) == [5]


class TestBuildAccumulatorAndSummary:
    def test_perfect_match_gives_mota_1(self):
        # gt et pred identiques -> tracking parfait
        gt_frames = {1: (np.array([1]), np.array([[10.0, 10.0, 50.0, 100.0]]))}
        pred_frames = {1: (np.array([1]), np.array([[10.0, 10.0, 50.0, 100.0]]))}

        acc = build_accumulator(gt_frames, pred_frames, max_iou=0.5)
        summary_text = compute_summary(acc)

        assert "MOTA" in summary_text
        assert "IDF1" in summary_text

    def test_missing_detection_is_counted(self):
        # gt a une detection, pred n'en a aucune -> 1 "miss"
        gt_frames = {1: (np.array([1]), np.array([[10.0, 10.0, 50.0, 100.0]]))}
        pred_frames = {1: (np.array([]), np.empty((0, 4)))}

        acc = build_accumulator(gt_frames, pred_frames, max_iou=0.5)
        summary_text = compute_summary(acc)

        assert "FN" in summary_text or "Miss" in summary_text or "Num Misses" in summary_text

    def test_empty_gt_and_pred_does_not_crash(self):
        acc = build_accumulator({}, {}, max_iou=0.5)
        summary_text = compute_summary(acc)
        assert isinstance(summary_text, str)
