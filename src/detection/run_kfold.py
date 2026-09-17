"""Effectue la validation croisée 7-fold sur le pool MOT17 converti.

Prérequis : avoir lancé `build_dataset.py --mode pool`

Pour chaque vidéo de base (7 au total), ce script :
1. Prépare un fold (symlinks train/val via src/detection/kfold.py)
2. Lance un entrainement via src/detection/train.py (memes hyperparametres
   pour tous les folds)
3. Lit le meilleur mAP50-95 atteint dans results.csv
Puis agrège les 7 résultats en moyenne +/- ecart-type.

Usage :
    python -m src.detection.run_kfold \\
        --pool-dir /kaggle/working/mot-reid-pipeline/data/processed/mot17_yolo_pool \\
        --model yolov8n.pt --freeze 10 --lr0 0.0003 --epochs 80 --patience 20 --device 0
"""
from __future__ import annotations
 
import argparse
import csv
import subprocess
import sys
from pathlib import Path
 
from src.detection.kfold import list_base_sequences, prepare_fold, read_manifest
 
 
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pool-dir", type=Path, required=True, help="Dossier du pool (contient manifest.csv)")
    parser.add_argument("--fold-root", type=Path, default=None, help="Ou ecrire les folds (defaut: a cote du pool)")
    parser.add_argument("--model", default="yolo26n.pt", help="yolo*.pt ou rtdetr-*.pt")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--freeze", type=int, default=None, help="Defaut auto selon le modele (voir train.py) si non fourni")
    parser.add_argument("--lr0", type=float, default=None, help="Defaut auto selon le modele (voir train.py) si non fourni")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=None, help="Defaut auto selon le modele (voir train.py) si non fourni")
    parser.add_argument("--device", default="0")
    parser.add_argument(
        "--domain-augment", action="store_true",
        help="Propage --domain-augment a chaque fold (voir train.py)",
    )
    parser.add_argument("--summary-csv", type=Path, default=None)
    args = parser.parse_args()

    if args.summary_csv is None:
        suffix = "rtdetr" if "rtdetr" in args.model.lower() else args.model.replace(".pt", "")
        args.summary_csv = Path(f"kfold_summary_{suffix}.csv")

    return args
 
 
def best_metrics_from_results_csv(results_csv: Path) -> dict[str, float]:
    """Lit results.csv d'ultralytics, retourne la ligne au meilleur mAP50-95."""
    with results_csv.open(newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
 
    map_col = "metrics/mAP50-95(B)"
    best_row = max(rows, key=lambda r: float(r[map_col]))
    return {
        "epoch": int(float(best_row["epoch"])),
        "precision": float(best_row["metrics/precision(B)"]),
        "recall": float(best_row["metrics/recall(B)"]),
        "mAP50": float(best_row["metrics/mAP50(B)"]),
        "mAP50-95": float(best_row[map_col]),
    }
 
 
def run_fold(
    held_out_base: str,
    pool_dir: Path,
    fold_root: Path,
    manifest_rows: list[tuple[str, str]],
    args: argparse.Namespace,
) -> dict[str, float]:
    fold_dir = fold_root / f"fold_{held_out_base}"
    n_train, n_val = prepare_fold(pool_dir, manifest_rows, held_out_base, fold_dir)
    print(f"\n=== Fold {held_out_base} : {n_train} train / {n_val} val ===")
 
    run_name = f"kfold_{args.model.replace('.pt', '')}_{held_out_base}"
    cmd = [
        sys.executable, "-m", "src.detection.train",
        "--data", str(fold_dir / "dataset.yaml"),
        "--model", args.model,
        "--epochs", str(args.epochs),
        "--patience", str(args.patience),
        "--imgsz", str(args.imgsz),
        "--device", args.device,
        "--name", run_name,
    ]
    if args.freeze is not None:
        cmd += ["--freeze", str(args.freeze)]
    if args.lr0 is not None:
        cmd += ["--lr0", str(args.lr0)]
    if args.batch is not None:
        cmd += ["--batch", str(args.batch)]
    if args.domain_augment:
        cmd.append("--domain-augment")
    subprocess.run(cmd, check=True)
 
    results_csv = Path("runs") / "detect" / run_name / "results.csv"
    metrics = best_metrics_from_results_csv(results_csv)
    metrics["held_out"] = held_out_base
    return metrics
 
 
def main() -> None:
    args = parse_args()
    manifest_rows = read_manifest(args.pool_dir / "manifest.csv")
    base_names = list_base_sequences(manifest_rows)
    fold_root = args.fold_root or (args.pool_dir.parent / "kfold_runs")
 
    print(f"{len(base_names)} folds a executer : {base_names}")
 
    all_metrics = []
    for held_out in base_names:
        metrics = run_fold(held_out, args.pool_dir, fold_root, manifest_rows, args)
        all_metrics.append(metrics)
        print(f"  -> mAP50-95 = {metrics['mAP50-95']:.3f} (epoch {metrics['epoch']})")
 
    # Agregation
    map_values = [m["mAP50-95"] for m in all_metrics]
    mean_map = sum(map_values) / len(map_values)
    variance = sum((x - mean_map) ** 2 for x in map_values) / len(map_values)
    std_map = variance ** 0.5
 
    print(f"\n{'=' * 50}")
    print(f"Resultat final ({args.model}, {len(base_names)} folds) :")
    print(f"  mAP50-95 = {mean_map:.3f} +/- {std_map:.3f}")
    print(f"  min={min(map_values):.3f}  max={max(map_values):.3f}")
    print(f"{'=' * 50}")
 
    with args.summary_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["held_out", "epoch", "precision", "recall", "mAP50", "mAP50-95"])
        writer.writeheader()
        writer.writerows(all_metrics)
    print(f"\nDetail par fold ecrit dans {args.summary_csv}")
 
 
if __name__ == "__main__":
    main()