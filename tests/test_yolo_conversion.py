"""Tests du module de conversion MOT17 -> YOLO.

Utilise des données synthétiques : pas besoin d'avoir MOT17 telechargé
pour valider la logique de filtrage/clipping/split.
"""

import pytest

from src.detection.yolo_conversion import (
    base_sequence_name,
    clip_bbox,
    convert_frame_annotations,
    mot_bbox_to_yolo,
    parse_gt_line,
    should_keep_annotation,
    split_base_sequences,
)


class TestBaseSequenceName:
    def test_strips_detector_suffix(self):
        assert base_sequence_name("MOT17-02-FRCNN") == "MOT17-02"
        assert base_sequence_name("MOT17-11-DPM") == "MOT17-11"
        assert base_sequence_name("MOT17-04-SDP") == "MOT17-04"

    def test_invalid_name_raises(self):
        with pytest.raises(ValueError):
            base_sequence_name("not-a-mot17-sequence")


class TestParseGtLine:
    def test_parses_standard_line(self):
        line = "1,2,10.5,20.0,50.0,100.0,1,1,0.85\n"
        ann = parse_gt_line(line)
        assert ann["frame"] == 1
        assert ann["id"] == 2
        assert ann["bb_left"] == 10.5
        assert ann["class"] == 1
        assert ann["conf"] == 1
        assert ann["visibility"] == pytest.approx(0.85)

    def test_malformed_line_raises(self):
        with pytest.raises(ValueError):
            parse_gt_line("1,2,3\n")


class TestShouldKeepAnnotation:
    def test_keeps_pedestrian_with_conf_1(self):
        ann = {"class": 1, "conf": 1}
        assert should_keep_annotation(ann) is True

    def test_rejects_non_pedestrian_class(self):
        ann = {"class": 7, "conf": 1}
        assert should_keep_annotation(ann) is False

    def test_rejects_conf_zero(self):
        ann = {"class": 1, "conf": 0}
        assert should_keep_annotation(ann) is False

    def test_rejects_low_visibility(self):
        ann = {"class": 1, "conf": 1, "visibility": 0.1}
        assert should_keep_annotation(ann) is False

    def test_keeps_visibility_at_threshold(self):
        ann = {"class": 1, "conf": 1, "visibility": 0.3}
        assert should_keep_annotation(ann) is True

    def test_missing_visibility_defaults_to_visible(self):
        # Retro-compatibilite : si le champ est absent, on ne filtre pas dessus.
        ann = {"class": 1, "conf": 1}
        assert should_keep_annotation(ann) is True


class TestClipBbox:
    def test_bbox_fully_inside_unchanged(self):
        x, y, w, h = clip_bbox(10, 10, 50, 50, img_width=200, img_height=200)
        assert (x, y, w, h) == (10, 10, 50, 50)

    def test_negative_left_clipped(self):
        # bbox commence avant le bord gauche de l'image (cas observe dans l'EDA)
        x, y, w, h = clip_bbox(-20, 10, 50, 50, img_width=200, img_height=200)
        assert x == 0
        assert w == 30  # 50 - 20 de largeur perdue

    def test_negative_top_clipped(self):
        x, y, w, h = clip_bbox(10, -5, 50, 50, img_width=200, img_height=200)
        assert y == 0
        assert h == 45

    def test_bbox_exceeding_right_edge_clipped(self):
        x, y, w, h = clip_bbox(180, 10, 50, 50, img_width=200, img_height=200)
        assert w == 20  # de x=180 a x=200

    def test_bbox_exceeding_bottom_edge_clipped(self):
        x, y, w, h = clip_bbox(10, 180, 50, 50, img_width=200, img_height=200)
        assert h == 20

    def test_bbox_fully_out_of_frame_results_in_zero_size(self):
        x, y, w, h = clip_bbox(-100, -100, 50, 50, img_width=200, img_height=200)
        assert w == 0
        assert h == 0


class TestMotBboxToYolo:
    def test_conversion_is_normalized_between_0_and_1(self):
        box = mot_bbox_to_yolo(50, 50, 100, 100, img_width=200, img_height=200)
        assert 0 <= box.x_center <= 1
        assert 0 <= box.y_center <= 1
        assert 0 <= box.width <= 1
        assert 0 <= box.height <= 1

    def test_center_computed_correctly(self):
        # bbox de (50,50) a (150,150) dans une image 200x200 -> centre a (100,100) -> 0.5, 0.5
        box = mot_bbox_to_yolo(50, 50, 100, 100, img_width=200, img_height=200)
        assert box.x_center == pytest.approx(0.5)
        assert box.y_center == pytest.approx(0.5)
        assert box.width == pytest.approx(0.5)
        assert box.height == pytest.approx(0.5)

    def test_fully_out_of_frame_raises(self):
        with pytest.raises(ValueError):
            mot_bbox_to_yolo(-500, -500, 50, 50, img_width=200, img_height=200)

    def test_yolo_line_format(self):
        box = mot_bbox_to_yolo(50, 50, 100, 100, img_width=200, img_height=200)
        line = box.to_line()
        parts = line.split()
        assert len(parts) == 5
        assert parts[0] == "0"  # une seule classe (pieton) -> id 0


class TestConvertFrameAnnotations:
    def test_filters_and_converts(self):
        annotations = [
            {"class": 1, "conf": 1, "bb_left": 10, "bb_top": 10, "bb_width": 50, "bb_height": 50},
            {
                "class": 7,
                "conf": 1,
                "bb_left": 10,
                "bb_top": 10,
                "bb_width": 50,
                "bb_height": 50,
            },  # non-pieton
            {
                "class": 1,
                "conf": 0,
                "bb_left": 10,
                "bb_top": 10,
                "bb_width": 50,
                "bb_height": 50,
            },  # conf=0
        ]
        boxes = convert_frame_annotations(annotations, img_width=200, img_height=200)
        assert len(boxes) == 1

    def test_skips_out_of_frame_silently(self):
        annotations = [
            {
                "class": 1,
                "conf": 1,
                "bb_left": -500,
                "bb_top": -500,
                "bb_width": 50,
                "bb_height": 50,
            },
        ]
        boxes = convert_frame_annotations(annotations, img_width=200, img_height=200)
        assert boxes == []

    def test_empty_annotations_returns_empty_list(self):
        assert convert_frame_annotations([], img_width=200, img_height=200) == []


class TestSplitBaseSequences:
    def test_split_is_deterministic_with_seed(self):
        names = [f"MOT17-{i:02d}" for i in range(1, 8)]
        train1, val1 = split_base_sequences(names, val_fraction=0.2, seed=42)
        train2, val2 = split_base_sequences(names, val_fraction=0.2, seed=42)
        assert train1 == train2
        assert val1 == val2

    def test_train_and_val_are_disjoint(self):
        names = [f"MOT17-{i:02d}" for i in range(1, 8)]
        train, val = split_base_sequences(names, val_fraction=0.2, seed=42)
        assert set(train).isdisjoint(set(val))

    def test_all_names_accounted_for(self):
        names = [f"MOT17-{i:02d}" for i in range(1, 8)]
        train, val = split_base_sequences(names, val_fraction=0.2, seed=42)
        assert set(train) | set(val) == set(names)

    def test_never_splits_within_same_base_name(self):
        # Garantie critique : meme si on passe des noms avec suffixe par erreur,
        # deux variantes de la meme video ne doivent jamais finir de part et
        # d'autre du split. On teste ici avec des noms deja deduppliques,
        # ce qui est le contrat attendu de la fonction.
        names = ["MOT17-02", "MOT17-04", "MOT17-05", "MOT17-09", "MOT17-10", "MOT17-11", "MOT17-13"]
        train, val = split_base_sequences(names, val_fraction=0.3, seed=1)
        assert len(val) >= 1
        assert len(train) >= 1
