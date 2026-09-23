"""Tests de count_gt_identities."""

import numpy as np

from src.tracking.count_gt_identities import count_unique_identities


class TestCountUniqueIdentities:
    def test_counts_distinct_ids_across_frames(self):
        gt_frames = {
            1: (np.array([1, 2, 3]), np.empty((3, 4))),
            2: (np.array([1, 2]), np.empty((2, 4))),  # 3 disparait
            3: (np.array([1, 4]), np.empty((2, 4))),  # 4 apparait
        }
        assert count_unique_identities(gt_frames) == 4  # {1,2,3,4}

    def test_empty_dict_returns_zero(self):
        assert count_unique_identities({}) == 0

    def test_single_frame(self):
        gt_frames = {1: (np.array([1, 2, 3]), np.empty((3, 4)))}
        assert count_unique_identities(gt_frames) == 3

    def test_repeated_id_across_frames_counted_once(self):
        gt_frames = {
            1: (np.array([1]), np.empty((1, 4))),
            2: (np.array([1]), np.empty((1, 4))),
            3: (np.array([1]), np.empty((1, 4))),
        }
        assert count_unique_identities(gt_frames) == 1
