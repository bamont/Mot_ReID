"""Tests de l'echantillonnage de triplets, avec un index synthétique."""

import random
from pathlib import Path

import pytest

from src.reid.triplet_sampling import generate_triplets, sample_triplet

# Trois identités, la premiere avec 2 caméras (utile pour tester le cross-caméra),
# la deuxieme mono-caméra (utile pour tester le fallback), la troisieme sert de negatif.
INDEX = {
    1: {
        1: [Path("id1_cam1_a.jpg"), Path("id1_cam1_b.jpg")],
        2: [Path("id1_cam2_a.jpg")],
    },
    2: {1: [Path("id2_cam1_a.jpg"), Path("id2_cam1_b.jpg")]},  # mono-caméra
    3: {1: [Path("id3_cam1_a.jpg")]},
}


class TestSampleTriplet:
    def test_anchor_and_positive_share_identity(self):
        rng = random.Random(0)
        for _ in range(50):
            t = sample_triplet(rng, INDEX)
            assert t.anchor != t.positive
            assert t.anchor_id != t.negative_id

    def test_negative_is_different_identity(self):
        rng = random.Random(0)
        for _ in range(50):
            t = sample_triplet(rng, INDEX)
            assert t.negative not in {p for cams in INDEX[t.anchor_id].values() for p in cams}

    def test_prefers_cross_camera_positive_when_available(self):
        # id=1 a 2 caméras : sur suffisamment de tirages avec anchor_id=1, on doit voir
        # le positif venir de cam2 au moins une fois si l'ancre vient de cam1.
        rng = random.Random(1)
        seen_cross_camera = False
        for _ in range(200):
            t = sample_triplet(rng, INDEX, prefer_cross_camera=True)
            if t.anchor_id == 1 and t.anchor in INDEX[1][1] and t.positive in INDEX[1][2]:
                seen_cross_camera = True
                break
        assert seen_cross_camera

    def test_mono_camera_identity_falls_back_to_same_camera(self):
        # id=2 n'a qu'une caméra : le positif DOIT venir de la meme caméra, different fichier.
        mono_camera_index = {2: INDEX[2], 3: INDEX[3]}
        rng = random.Random(0)
        for _ in range(50):
            t = sample_triplet(rng, mono_camera_index, prefer_cross_camera=True)
            assert t.anchor_id == 2
            assert t.positive in INDEX[2][1]
            assert t.positive != t.anchor

    def test_raises_if_no_identity_has_two_images(self):
        singleton_index = {1: {1: [Path("a.jpg")]}, 2: {1: [Path("b.jpg")]}}
        with pytest.raises(ValueError):
            sample_triplet(random.Random(0), singleton_index)

    def test_raises_if_fewer_than_two_identities(self):
        with pytest.raises(ValueError):
            sample_triplet(random.Random(0), {1: INDEX[1]})


class TestGenerateTriplets:
    def test_returns_requested_count(self):
        triplets = generate_triplets(INDEX, n_triplets=25, seed=42)
        assert len(triplets) == 25

    def test_deterministic_given_seed(self):
        a = generate_triplets(INDEX, n_triplets=10, seed=123)
        b = generate_triplets(INDEX, n_triplets=10, seed=123)
        assert a == b

    def test_different_seeds_differ(self):
        a = generate_triplets(INDEX, n_triplets=10, seed=1)
        b = generate_triplets(INDEX, n_triplets=10, seed=2)
        assert a != b

    def test_negative_n_raises(self):
        with pytest.raises(ValueError):
            generate_triplets(INDEX, n_triplets=-1, seed=0)
