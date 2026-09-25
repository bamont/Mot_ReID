"""Entraine l'encodeur de ré-identification sur Market-1501 (triplet loss).

Usage:
    poetry run python -m src.reid.train
    poetry run python -m src.reid.train --mining random --epochs 30 --batch-size 32
    poetry run python -m src.reid.train --mining batch-hard --p 16 --k 4 --epochs 30

Deux strategies de mining disponibles (``--mining``) :
    - "batch-hard" (par defaut) : batches de P identites x K images, positif/négatif les
      plus durs minés à l'interieur du batch (Hermans et al. 2017).
    - "random" : un triplet different par item, négatif tire au hasard (implementation
      initiale, conservée pour comparaison empirique -- cf. TripletMarket1501Dataset).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from src.config import REID_TRIPLETS_DIR
from src.reid.dataset import TripletMarket1501Dataset
from src.reid.market1501 import flatten_identity_index, load_identity_index_json
from src.reid.model import build_model, build_transforms
from src.reid.pk_sampling import (
    PKBatchSampler,
    PKImageDataset,
    batch_hard_triplet_loss,
    mine_batch_hard,
)

# --------------------------------------------------------------------------------------
# Mode "random" : un triplet aleatoire par item (TripletMarket1501Dataset).
# --------------------------------------------------------------------------------------


def train_one_epoch_random(
    model: torch.nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: torch.nn.Module,
    device: torch.device,
) -> float:
    model.train()
    total_loss = 0.0
    n_batches = 0
    for batch in loader:
        anchor = batch["anchor"].to(device)
        positive = batch["positive"].to(device)
        negative = batch["negative"].to(device)

        optimizer.zero_grad()
        loss = criterion(model(anchor), model(positive), model(negative))
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        n_batches += 1
    return total_loss / max(n_batches, 1)


@torch.no_grad()
def validate_random(
    model: torch.nn.Module,
    loader: DataLoader,
    criterion: torch.nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    """Retourne (loss moyenne, fraction de triplets qui respectent la marge sur ce batch).
    """
    model.eval()
    total_loss = 0.0
    n_batches = 0
    n_satisfied = 0
    n_total = 0
    for batch in loader:
        anchor = batch["anchor"].to(device)
        positive = batch["positive"].to(device)
        negative = batch["negative"].to(device)

        emb_a, emb_p, emb_n = model(anchor), model(positive), model(negative)
        loss = criterion(emb_a, emb_p, emb_n)
        total_loss += loss.item()
        n_batches += 1

        d_pos = (emb_a - emb_p).pow(2).sum(dim=1)
        d_neg = (emb_a - emb_n).pow(2).sum(dim=1)
        n_satisfied += (d_pos + criterion.margin < d_neg).sum().item()
        n_total += anchor.size(0)

    return total_loss / max(n_batches, 1), n_satisfied / max(n_total, 1)


# --------------------------------------------------------------------------------------
# Mode "batch-hard" : batches P identites x K images, mining a l'interieur du batch.
# --------------------------------------------------------------------------------------


def train_one_epoch_batch_hard(
    model: torch.nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    margin: float,
    device: torch.device,
) -> float:
    model.train()
    total_loss = 0.0
    n_batches = 0
    for images, pids in loader:
        images, pids = images.to(device), pids.to(device)

        optimizer.zero_grad()
        loss = batch_hard_triplet_loss(model(images), pids, margin=margin)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        n_batches += 1
    return total_loss / max(n_batches, 1)


@torch.no_grad()
def validate_batch_hard(
    model: torch.nn.Module,
    loader: DataLoader,
    margin: float,
    device: torch.device,
) -> tuple[float, float]:
    model.eval()
    total_loss = 0.0
    n_batches = 0
    n_satisfied = 0
    n_total = 0
    for images, pids in loader:
        images, pids = images.to(device), pids.to(device)

        embeddings = model(images)
        hardest_positive, hardest_negative = mine_batch_hard(embeddings, pids)
        loss = torch.relu(hardest_positive - hardest_negative + margin).mean()
        total_loss += loss.item()
        n_batches += 1

        n_satisfied += (hardest_positive + margin < hardest_negative).sum().item()
        n_total += len(pids)

    return total_loss / max(n_batches, 1), n_satisfied / max(n_total, 1)


# --------------------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--mining", choices=["batch-hard", "random"], default="batch-hard")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--margin", type=float, default=0.3)
    parser.add_argument("--embedding-dim", type=int, default=512)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("runs/reid"))
    parser.add_argument(
        "--no-pretrained",
        action="store_false",
        dest="pretrained",
        help="Backbone ResNet50 initialisé aléatoirement plutôt que pée-entrainé ImageNet.",
    )

    # --mining batch-hard
    parser.add_argument("--p", type=int, default=16, help="Identités par batch (mode batch-hard)")
    parser.add_argument("--k", type=int, default=4, help="Images par identité et par batch")
    parser.add_argument(
        "--batches-per-epoch",
        type=int,
        default=None,
        help="Defaut : 4x (nb identités train / p)",
    )
    parser.add_argument("--val-batches", type=int, default=20)

    # --mining random
    parser.add_argument("--batch-size", type=int, default=32, help="Mode random uniquement")
    parser.add_argument(
        "--triplets-per-epoch",
        type=int,
        default=None,
        help="Mode random uniquement. Defaut : 4x le nombre d'identités train",
    )
    parser.add_argument("--val-triplets", type=int, default=2000, help="Mode random uniquement")

    return parser.parse_args()


def _run_batch_hard(args, train_index, val_index, model, optimizer, device) -> None:
    train_flat = flatten_identity_index(train_index)
    val_flat = flatten_identity_index(val_index)
    batches_per_epoch = args.batches_per_epoch or max(1, (len(train_flat) * 4) // args.p)

    train_dataset = PKImageDataset(train_flat, transform=build_transforms(train=True))
    val_dataset = PKImageDataset(val_flat, transform=build_transforms(train=False))
    train_sampler = PKBatchSampler(
        train_dataset, p=args.p, k=args.k, batches_per_epoch=batches_per_epoch, seed=args.seed
    )
    val_sampler = PKBatchSampler(
        val_dataset,
        p=min(args.p, len(val_flat)),
        k=args.k,
        batches_per_epoch=args.val_batches,
        seed=args.seed,
    )
    train_loader = DataLoader(
        train_dataset, batch_sampler=train_sampler, num_workers=args.num_workers
    )
    val_loader = DataLoader(val_dataset, batch_sampler=val_sampler, num_workers=args.num_workers)

    print(
        f"Train : {len(train_flat)} identités, {batches_per_epoch} batches/epoque "
        f"(p={args.p}, k={args.k}, taille batch={args.p * args.k}) | "
        f"Val : {len(val_flat)} identités, {args.val_batches} batches | device={device}"
    )

    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    best_val_loss = float("inf")

    for epoch in range(args.epochs):
        train_sampler.set_epoch(epoch)
        train_loss = train_one_epoch_batch_hard(model, train_loader, optimizer, args.margin, device)
        val_loss, val_margin_ok = validate_batch_hard(model, val_loader, args.margin, device)
        print(
            f"epoch {epoch + 1}/{args.epochs}  "
            f"train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  "
            f"val_margin_ok={val_margin_ok:.1%}"
        )

        torch.save(model.state_dict(), args.checkpoint_dir / "last.pt")
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), args.checkpoint_dir / "best.pt")


def _run_random(args, train_index, val_index, model, optimizer, device) -> None:
    criterion = torch.nn.TripletMarginLoss(margin=args.margin, p=2)
    triplets_per_epoch = args.triplets_per_epoch or len(train_index) * 4

    train_dataset = TripletMarket1501Dataset(
        train_index,
        length=triplets_per_epoch,
        transform=build_transforms(train=True),
        seed=args.seed,
    )
    val_dataset = TripletMarket1501Dataset(
        val_index, length=args.val_triplets, transform=build_transforms(train=False), seed=args.seed
    )
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, num_workers=args.num_workers, drop_last=True
    )
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, num_workers=args.num_workers)

    print(
        f"Train : {len(train_index)} identites, {triplets_per_epoch} triplets/epoque | "
        f"Val : {len(val_index)} identites, {args.val_triplets} triplets | device={device}"
    )

    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    best_val_loss = float("inf")

    for epoch in range(args.epochs):
        train_dataset.set_epoch(epoch)
        train_loss = train_one_epoch_random(model, train_loader, optimizer, criterion, device)
        val_loss, val_margin_ok = validate_random(model, val_loader, criterion, device)
        print(
            f"epoch {epoch + 1}/{args.epochs}  "
            f"train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  "
            f"val_margin_ok={val_margin_ok:.1%}"
        )

        torch.save(model.state_dict(), args.checkpoint_dir / "last.pt")
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), args.checkpoint_dir / "best.pt")


def main() -> None:
    args = parse_args()

    train_index = load_identity_index_json(REID_TRIPLETS_DIR / "train_identities.json")
    val_index = load_identity_index_json(REID_TRIPLETS_DIR / "val_identities.json")

    device = torch.device(args.device)
    model = build_model(embedding_dim=args.embedding_dim, pretrained=args.pretrained).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    if args.mining == "batch-hard":
        _run_batch_hard(args, train_index, val_index, model, optimizer, device)
    else:
        _run_random(args, train_index, val_index, model, optimizer, device)

    print(f"\nTermine. Checkpoints dans {args.checkpoint_dir}")


if __name__ == "__main__":
    main()
