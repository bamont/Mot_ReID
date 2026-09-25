"""Evalue l'encodeur ré-ID entraine sur le protocole officiel Market-1501 (mAP / CMC).

Usage:
    poetry run python -m src.reid.evaluate --checkpoint runs/reid/best.pt
    poetry run python -m src.reid.evaluate --checkpoint runs/reid/best.pt --device cuda
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.config import MARKET1501_DIR
from src.reid.dataset import ImageListDataset
from src.reid.market1501 import ParsedImage, list_images
from src.reid.model import build_model, build_transforms

QUERY_DIR = MARKET1501_DIR / "query"
GALLERY_DIR = MARKET1501_DIR / "bounding_box_test"


def compute_distance_matrix(
    query_embeddings: torch.Tensor, gallery_embeddings: torch.Tensor
) -> np.ndarray:
    """Distance euclidienne au carré entre embeddings L2-normalisés.
    Pour des vecteurs unitaires, ||a-b||^2 = 2 - 2*cos(a,b).
    """
    similarity = query_embeddings @ gallery_embeddings.T
    distance = (2 - 2 * similarity).clamp(min=0)
    return distance.cpu().numpy()


def compute_cmc_map(
    distmat: np.ndarray,
    q_pids: np.ndarray,
    g_pids: np.ndarray,
    q_camids: np.ndarray,
    g_camids: np.ndarray,
    max_rank: int = 50,
) -> tuple[np.ndarray, float]:
    """Courbe CMC (Cumulative Matching Characteristic) et mAP, protocole Market-1501.
    """
    num_q, num_g = distmat.shape
    max_rank = min(max_rank, num_g)
    order = np.argsort(distmat, axis=1)
    matches = (g_pids[order] == q_pids[:, None]).astype(np.int32)

    all_cmc = []
    all_ap = []
    n_valid_queries = 0

    for q_idx in range(num_q):
        q_pid, q_camid = q_pids[q_idx], q_camids[q_idx]
        ranked = order[q_idx]

        remove = (g_pids[ranked] == q_pid) & (g_camids[ranked] == q_camid)
        keep = ~remove

        raw_cmc = matches[q_idx][keep]
        if not raw_cmc.any():
            continue

        cmc = raw_cmc.cumsum()
        cmc[cmc > 1] = 1
        all_cmc.append(cmc[:max_rank])
        n_valid_queries += 1

        n_relevant = raw_cmc.sum()
        precision_at_k = raw_cmc.cumsum() / (np.arange(len(raw_cmc)) + 1)
        ap = (precision_at_k * raw_cmc).sum() / n_relevant
        all_ap.append(ap)

    if n_valid_queries == 0:
        raise ValueError(
            "Aucune query évaluable : vérifier que les identités de query existent bien "
            "dans la galerie sur au moins une caméra différente."
        )

    cmc = np.stack(all_cmc).astype(np.float64).sum(axis=0) / n_valid_queries
    mean_ap = float(np.mean(all_ap))
    return cmc, mean_ap


@torch.no_grad()
def extract_embeddings(
    model: torch.nn.Module,
    paths: list[Path],
    transform,
    device: torch.device,
    batch_size: int = 128,
    num_workers: int = 4,
) -> torch.Tensor:
    model.eval()
    dataset = ImageListDataset(paths, transform=transform)
    loader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers)
    embeddings = [model(batch.to(device)).cpu() for batch in loader]
    return torch.cat(embeddings, dim=0)


def _pids_and_camids(images: list[ParsedImage]) -> tuple[np.ndarray, np.ndarray]:
    return (
        np.array([im.pid for im in images]),
        np.array([im.camera for im in images]),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--embedding-dim", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--max-rank", type=int, default=50)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    query_images = list_images(QUERY_DIR)
    gallery_images = list_images(GALLERY_DIR)  # exclut deja pid 0 (junk) et -1 (distracteurs)
    if not query_images or not gallery_images:
        raise FileNotFoundError(
            f"query/ ou bounding_box_test/ introuvable ou vide sous {MARKET1501_DIR}."
        )

    device = torch.device(args.device)
    model = build_model(embedding_dim=args.embedding_dim, pretrained=False).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))

    transform = build_transforms(train=False)
    print(f"Extraction des embeddings ({len(query_images)} query, {len(gallery_images)} galerie)...")
    q_embeddings = extract_embeddings(
        model, [im.path for im in query_images], transform, device, args.batch_size, args.num_workers
    )
    g_embeddings = extract_embeddings(
        model, [im.path for im in gallery_images], transform, device, args.batch_size, args.num_workers
    )

    distmat = compute_distance_matrix(q_embeddings, g_embeddings)
    q_pids, q_camids = _pids_and_camids(query_images)
    g_pids, g_camids = _pids_and_camids(gallery_images)

    cmc, mean_ap = compute_cmc_map(distmat, q_pids, g_pids, q_camids, g_camids, args.max_rank)

    print(f"\nmAP: {mean_ap:.1%}")
    for r in (1, 5, 10, 20):
        if r <= len(cmc):
            print(f"Rank-{r}: {cmc[r - 1]:.1%}")


if __name__ == "__main__":
    main()
