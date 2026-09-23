"""Tests du module de parsing/indexation Market-1501.

Utilise des fichiers vides avec des noms synthétiques : pas besoin d'avoir
Market-1501 téléchargé pour valider la logique de parsing/filtrage/split.
"""

import pytest

from src.reid.market1501 import (
    build_identity_index,
    count_images,
    identities_with_min_images,
    is_valid_identity,
    parse_filename,
    split_identities,
    subindex,
)


class TestParseFilename:
    def test_parses_standard_name(self):
        parsed = parse_filename("0002_c1s1_000451_03.jpg")
        assert parsed.pid == 2
        assert parsed.camera == 1
        assert parsed.sequence == 1
        assert parsed.frame == 451
        assert parsed.box == 3

    def test_parses_junk_pid(self):
        assert parse_filename("0000_c3s2_012345_01.jpg").pid == 0

    def test_parses_distractor_pid(self):
        assert parse_filename("-1_c3s2_012345_01.jpg").pid == -1

    def test_invalid_name_raises(self):
        with pytest.raises(ValueError):
            parse_filename("not_a_market1501_file.jpg")

    def test_rejects_wrong_extension(self):
        with pytest.raises(ValueError):
            parse_filename("0002_c1s1_000451_03.txt")


class TestIsValidIdentity:
    def test_rejects_junk(self):
        assert is_valid_identity(0) is False

    def test_rejects_distractor(self):
        assert is_valid_identity(-1) is False

    def test_accepts_real_identity(self):
        assert is_valid_identity(2) is True


class TestBuildIdentityIndex:
    def _touch(self, tmp_path, names):
        for name in names:
            (tmp_path / name).touch()
        return tmp_path

    def test_groups_by_identity_and_camera(self, tmp_path):
        self._touch(
            tmp_path,
            [
                "0002_c1s1_000451_03.jpg",
                "0002_c1s1_000452_01.jpg",  # meme id, même caméra
                "0002_c2s1_000010_01.jpg",  # meme id, autre caméra
                "0007_c1s1_000001_01.jpg",  # autre id
            ],
        )
        index = build_identity_index(tmp_path)
        assert set(index) == {2, 7}
        assert len(index[2][1]) == 2
        assert len(index[2][2]) == 1
        assert len(index[7][1]) == 1

    def test_excludes_junk_and_distractors_by_default(self, tmp_path):
        self._touch(
            tmp_path,
            [
                "0000_c1s1_000001_01.jpg",
                "-1_c1s1_000002_01.jpg",
                "0002_c1s1_000451_03.jpg",
            ],
        )
        index = build_identity_index(tmp_path)
        assert set(index) == {2}

    def test_valid_only_false_keeps_everything(self, tmp_path):
        self._touch(tmp_path, ["0000_c1s1_000001_01.jpg", "0002_c1s1_000451_03.jpg"])
        index = build_identity_index(tmp_path, valid_only=False)
        assert set(index) == {0, 2}

    def test_ignores_non_matching_files(self, tmp_path):
        self._touch(tmp_path, ["Thumbs.db", "readme.txt", "0002_c1s1_000451_03.jpg"])
        index = build_identity_index(tmp_path)
        assert set(index) == {2}

    def test_count_images(self, tmp_path):
        self._touch(
            tmp_path,
            ["0002_c1s1_000451_03.jpg", "0002_c2s1_000010_01.jpg", "0007_c1s1_000001_01.jpg"],
        )
        assert count_images(build_identity_index(tmp_path)) == 3


class TestIdentitiesWithMinImages:
    def test_filters_singleton_identities(self):
        index = {
            1: {1: ["a.jpg", "b.jpg"]},  # 2 images -> éligible
            2: {1: ["c.jpg"]},  # 1 image -> pas éligible
        }
        assert identities_with_min_images(index, min_images=2) == [1]

    def test_counts_across_cameras(self):
        index = {1: {1: ["a.jpg"], 2: ["b.jpg"]}}  # 1 image par caméra, 2 au total
        assert identities_with_min_images(index, min_images=2) == [1]


class TestSplitIdentities:
    def test_split_sizes_match_ratio(self):
        ids = list(range(100))
        train, val = split_identities(ids, val_ratio=0.1, seed=42)
        assert len(val) == 10
        assert len(train) == 90

    def test_split_is_disjoint_and_covers_all_ids(self):
        ids = list(range(50))
        train, val = split_identities(ids, val_ratio=0.2, seed=1)
        assert set(train) & set(val) == set()
        assert set(train) | set(val) == set(ids)

    def test_split_is_deterministic_given_seed(self):
        ids = list(range(50))
        train_a, val_a = split_identities(ids, val_ratio=0.2, seed=7)
        train_b, val_b = split_identities(ids, val_ratio=0.2, seed=7)
        assert train_a == train_b
        assert val_a == val_b

    def test_different_seeds_give_different_splits(self):
        ids = list(range(50))
        _, val_a = split_identities(ids, val_ratio=0.2, seed=1)
        _, val_b = split_identities(ids, val_ratio=0.2, seed=2)
        assert val_a != val_b

    def test_invalid_ratio_raises(self):
        with pytest.raises(ValueError):
            split_identities([1, 2, 3], val_ratio=1.0)


class TestSubindex:
    def test_keeps_only_requested_identities(self):
        index = {1: {1: ["a.jpg"]}, 2: {1: ["b.jpg"]}, 3: {1: ["c.jpg"]}}
        assert set(subindex(index, [1, 3])) == {1, 3}
