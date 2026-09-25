"""Tests du module d'evaluation ré-ID (src/reid/evaluate.py).

Nom du fichier volontairement distinct de tests/test_evaluate.py (qui teste le module
de tracking, sans rapport) pour eviter toute collision.
"""

import numpy as np
import pytest
import torch

from src.reid.evaluate import compute_cmc_map, compute_distance_matrix


class TestComputeDistanceMatrix:
    def test_identical_normalized_vectors_have_zero_distance(self):
        v = torch.nn.functional.normalize(torch.tensor([[1.0, 2.0, 3.0]]), dim=1)
        dist = compute_distance_matrix(v, v)
        assert dist[0, 0] == pytest.approx(0.0, abs=1e-5)

    def test_orthogonal_normalized_vectors_have_distance_two(self):
        q = torch.tensor([[1.0, 0.0]])
        g = torch.tensor([[0.0, 1.0]])
        dist = compute_distance_matrix(q, g)
        assert dist[0, 0] == pytest.approx(2.0, abs=1e-5)

    def test_opposite_normalized_vectors_have_distance_four(self):
        q = torch.tensor([[1.0, 0.0]])
        g = torch.tensor([[-1.0, 0.0]])
        dist = compute_distance_matrix(q, g)
        assert dist[0, 0] == pytest.approx(4.0, abs=1e-5)

    def test_output_shape(self):
        q = torch.randn(5, 8)
        g = torch.randn(7, 8)
        q = torch.nn.functional.normalize(q, dim=1)
        g = torch.nn.functional.normalize(g, dim=1)
        assert compute_distance_matrix(q, g).shape == (5, 7)


class TestComputeCmcMap:
    def test_matches_hand_computed_toy_example(self):
        # 2 queries, 4 images de galerie -- calcul detaille dans la doc de la PR /
        # discussion : la query 1 est ignoree (sa seule occurrence en galerie est sur
        # sa propre camera), seule la query 0 compte dans le resultat final.
        distmat = np.array(
            [
                [0.1, 0.5, 0.3, 0.9],  # query 0
                [0.4, 0.2, 0.05, 0.6],  # query 1
            ]
        )
        g_pids = np.array([1, 1, 2, 3])
        g_camids = np.array([2, 3, 2, 2])
        q_pids = np.array([1, 2])
        q_camids = np.array([1, 2])

        cmc, mean_ap = compute_cmc_map(distmat, q_pids, g_pids, q_camids, g_camids, max_rank=4)

        # Query 0 : classement trie = [g0(pid1), g2(pid2), g1(pid1), g3(pid3)]
        # -> aucun retrait (aucune image de galerie n'est sur la camera 1 de la query)
        # -> hits = [1, 0, 1, 0] -> cmc cumule clippe = [1, 1, 1, 1]
        # -> AP = (1/1 + 2/3) / 2 = 0.8333...
        assert cmc == pytest.approx([1.0, 1.0, 1.0, 1.0])
        assert mean_ap == pytest.approx(0.8333333, abs=1e-5)

    def test_perfect_ranking_gives_map_one(self):
        # Chaque query trouve son match exact en rang 1, sur une camera differente.
        distmat = np.array([[0.0, 1.0], [1.0, 0.0]])
        g_pids = np.array([10, 20])
        g_camids = np.array([2, 2])
        q_pids = np.array([10, 20])
        q_camids = np.array([1, 1])

        cmc, mean_ap = compute_cmc_map(distmat, q_pids, g_pids, q_camids, g_camids)
        assert mean_ap == pytest.approx(1.0)
        assert cmc[0] == pytest.approx(1.0)

    def test_raises_when_no_query_is_evaluable(self):
        # La seule identite de galerie partage toujours la camera de sa query.
        distmat = np.array([[0.1]])
        g_pids = np.array([1])
        g_camids = np.array([1])
        q_pids = np.array([1])
        q_camids = np.array([1])

        with pytest.raises(ValueError):
            compute_cmc_map(distmat, q_pids, g_pids, q_camids, g_camids)

    def test_max_rank_truncates_cmc_length(self):
        distmat = np.array([[0.1, 0.2, 0.3, 0.4, 0.5]])
        g_pids = np.array([1, 2, 3, 4, 5])
        g_camids = np.array([2, 2, 2, 2, 2])
        q_pids = np.array([1])
        q_camids = np.array([1])

        cmc, _ = compute_cmc_map(distmat, q_pids, g_pids, q_camids, g_camids, max_rank=3)
        assert len(cmc) == 3
