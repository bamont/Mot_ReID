"""Tests du module de préparation des folds de validation croisée."""

import csv

import pytest

from src.detection.kfold import (
    assign_split,
    list_base_sequences,
    prepare_fold,
    read_manifest,
)

MANIFEST_ROWS = [
    ("MOT17-02-FRCNN_000001.jpg", "MOT17-02"),
    ("MOT17-02-FRCNN_000002.jpg", "MOT17-02"),
    ("MOT17-04-FRCNN_000001.jpg", "MOT17-04"),
    ("MOT17-05-FRCNN_000001.jpg", "MOT17-05"),
]


class TestReadManifest:
    def test_reads_csv_correctly(self, tmp_path):
        manifest_path = tmp_path / "manifest.csv"
        with manifest_path.open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["filename", "base_sequence"])
            writer.writerows(MANIFEST_ROWS)

        rows = read_manifest(manifest_path)
        assert rows == MANIFEST_ROWS


class TestListBaseSequences:
    def test_returns_unique_sorted_bases(self):
        bases = list_base_sequences(MANIFEST_ROWS)
        assert bases == ["MOT17-02", "MOT17-04", "MOT17-05"]


class TestAssignSplit:
    def test_held_out_base_goes_to_val(self):
        assignment = assign_split(MANIFEST_ROWS, held_out_base="MOT17-04")
        assert assignment["MOT17-04-FRCNN_000001.jpg"] == "val"

    def test_other_bases_go_to_train(self):
        assignment = assign_split(MANIFEST_ROWS, held_out_base="MOT17-04")
        assert assignment["MOT17-02-FRCNN_000001.jpg"] == "train"
        assert assignment["MOT17-02-FRCNN_000002.jpg"] == "train"
        assert assignment["MOT17-05-FRCNN_000001.jpg"] == "train"

    def test_every_file_assigned(self):
        assignment = assign_split(MANIFEST_ROWS, held_out_base="MOT17-02")
        assert len(assignment) == len(MANIFEST_ROWS)

    def test_different_held_out_gives_different_split(self):
        assignment_02 = assign_split(MANIFEST_ROWS, held_out_base="MOT17-02")
        assignment_04 = assign_split(MANIFEST_ROWS, held_out_base="MOT17-04")
        assert assignment_02 != assignment_04


class TestPrepareFold:
    @pytest.fixture
    def pool_dir(self, tmp_path):
        """Cree un pool minimal (images+labels) correspondant a MANIFEST_ROWS."""
        pool = tmp_path / "pool"
        (pool / "images").mkdir(parents=True)
        (pool / "labels").mkdir(parents=True)
        for filename, _ in MANIFEST_ROWS:
            (pool / "images" / filename).write_bytes(b"fake-jpg-content")
            label_name = filename.rsplit(".", 1)[0] + ".txt"
            (pool / "labels" / label_name).write_text("0 0.5 0.5 0.1 0.1\n")
        return pool

    def test_creates_expected_directory_structure(self, tmp_path, pool_dir):
        fold_dir = tmp_path / "fold_MOT17-04"
        prepare_fold(pool_dir, MANIFEST_ROWS, held_out_base="MOT17-04", fold_output_dir=fold_dir)

        assert (fold_dir / "images" / "train").is_dir()
        assert (fold_dir / "images" / "val").is_dir()
        assert (fold_dir / "labels" / "train").is_dir()
        assert (fold_dir / "labels" / "val").is_dir()

    def test_correct_counts_per_split(self, tmp_path, pool_dir):
        fold_dir = tmp_path / "fold_MOT17-04"
        n_train, n_val = prepare_fold(pool_dir, MANIFEST_ROWS, held_out_base="MOT17-04", fold_output_dir=fold_dir)

        assert n_train == 3  # MOT17-02 (x2) + MOT17-05
        assert n_val == 1  # MOT17-04

    def test_symlinks_point_to_real_content(self, tmp_path, pool_dir):
        fold_dir = tmp_path / "fold_MOT17-04"
        prepare_fold(pool_dir, MANIFEST_ROWS, held_out_base="MOT17-04", fold_output_dir=fold_dir)

        val_img = fold_dir / "images" / "val" / "MOT17-04-FRCNN_000001.jpg"
        assert val_img.is_symlink()
        assert val_img.read_bytes() == b"fake-jpg-content"

    def test_writes_dataset_yaml(self, tmp_path, pool_dir):
        fold_dir = tmp_path / "fold_MOT17-04"
        prepare_fold(pool_dir, MANIFEST_ROWS, held_out_base="MOT17-04", fold_output_dir=fold_dir)

        yaml_path = fold_dir / "dataset.yaml"
        assert yaml_path.exists()
        content = yaml_path.read_text()
        assert "train: images/train" in content
        assert "val: images/val" in content

    def test_different_fold_reuses_same_pool_without_conflict(self, tmp_path, pool_dir):
        # Deux folds differents dans des dossiers de sortie differents ne doivent pas interferer
        fold_02 = tmp_path / "fold_MOT17-02"
        fold_04 = tmp_path / "fold_MOT17-04"
        _, n_val_02 = prepare_fold(pool_dir, MANIFEST_ROWS, "MOT17-02", fold_02)
        _, n_val_04 = prepare_fold(pool_dir, MANIFEST_ROWS, "MOT17-04", fold_04)

        assert n_val_02 == 2  # les 2 frames MOT17-02
        assert n_val_04 == 1  # la frame MOT17-04