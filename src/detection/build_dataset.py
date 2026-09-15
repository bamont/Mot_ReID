from __future__ import annotations

"""Construit le dataset YOLO complet à partir de MOT17.

Usage:
    poetry run python -m src.detection.build_dataset

Produit :
    data/processed/mot17_yolo/
        images/{train,val}/*.jpg   (copies ou liens symboliques des frames)
        labels/{train,val}/*.txt    (annotations YOLO)
        dataset.yaml                 (config pour ultralytics)
"""

"""Construit le dataset YOLO a partir de MOT17.

Deux modes :

1. Mode "split" (par défaut) un seul split train/val fixe, aleatoire par video de base :
       poetry run python -m src.detection.build_dataset
   Produit :
       data/processed/mot17_yolo/
           images/{train,val}/*.jpg
           labels/{train,val}/*.txt
           dataset.yaml

2. Mode "pool" convertit les 7 videos SANS split, avec un manifeste
   (manifest.csv) qui indique la video de base de chaque frame. Sert de
   base pour la validation croisee (voir src/detection/kfold.py), afin de
   ne convertir/uploader les donnees qu'une seule fois plutot que 7 fois :
       poetry run python -m src.detection.build_dataset --mode pool
   Produit :
       data/processed/mot17_yolo_pool/
           images/*.jpg
           labels/*.txt
           manifest.csv   (colonnes: filename, base_sequence)
"""

import argparse
import configparser
import csv
import shutil
from pathlib import Path

from src.config import DATA_PROCESSED, MOT17_DIR
from src.detection.yolo_conversion import (
    base_sequence_name,
    convert_frame_annotations,
    parse_gt_line,
    split_base_sequences,
    write_yolo_label_file,
)

OUTPUT_DIR = DATA_PROCESSED / "mot17_yolo"
POOL_DIR = DATA_PROCESSED / "mot17_yolo_pool"


def read_seqinfo(seq_path: Path) -> tuple[int, int]:
    cfg = configparser.ConfigParser()
    cfg.read(seq_path / "seqinfo.ini")
    s = cfg["Sequence"]
    return int(s.get("imWidth")), int(s.get("imHeight"))


def load_annotations_by_frame(gt_path: Path) -> dict[int, list[dict]]:
    by_frame: dict[int, list[dict]] = {}
    with gt_path.open() as f:
        for line in f:
            if not line.strip():
                continue
            ann = parse_gt_line(line)
            by_frame.setdefault(int(ann["frame"]), []).append(ann)
    return by_frame


def process_sequence(seq_path: Path, images_out: Path, labels_out: Path) -> int:
    """Convertit une sequence complete, ecrit images+labels dans les dossiers donnes.

    Retourne le nombre de frames traitees.
    """
    img_width, img_height = read_seqinfo(seq_path)
    annotations_by_frame = load_annotations_by_frame(seq_path / "gt" / "gt.txt")

    images_out.mkdir(parents=True, exist_ok=True)
    labels_out.mkdir(parents=True, exist_ok=True)

    frame_paths = sorted((seq_path / "img1").glob("*.jpg"))
    n_processed = 0
    for frame_path in frame_paths:
        frame_num = int(frame_path.stem)
        raw_annotations = annotations_by_frame.get(frame_num, [])
        boxes = convert_frame_annotations(raw_annotations, img_width, img_height)

        # Prefixe par le nom de sequence pour eviter les collisions de noms
        # (chaque sequence recommence sa numerotation de frame a 000001).
        out_name = f"{seq_path.name}_{frame_path.stem}"
        shutil.copy(frame_path, images_out / f"{out_name}.jpg")
        write_yolo_label_file(boxes, labels_out / f"{out_name}.txt")
        n_processed += 1

    return n_processed


def pick_variant(all_sequences: list[Path], base_name: str) -> Path:
    """Choisit UNE variante (FRCNN par defaut) par video de base.

    Pas besoin des 3 (DPM/FRCNN/SDP) puisque gt.txt est identique entre
    variantes d'une meme video (verifie en EDA).
    """
    candidates = [s for s in all_sequences if s.name.startswith(base_name + "-")]
    preferred = next((s for s in candidates if s.name.endswith("-FRCNN")), None)
    return preferred or candidates[0]


def write_dataset_yaml(output_dir: Path) -> None:
    content = f"""# Genere par src/detection/build_dataset.py -- ne pas editer a la main
path: {output_dir.resolve()}
train: images/train
val: images/val
names:
  0: pedestrian
"""
    (output_dir / "dataset.yaml").write_text(content)


def build_split_dataset() -> None:
    """Mode historique : un seul split train/val fixe, aleatoire par video."""
    all_sequences = sorted(MOT17_DIR.glob("train/*"))
    base_names = sorted({base_sequence_name(s.name) for s in all_sequences})
    train_bases, val_bases = split_base_sequences(base_names)

    print(f"Split : {len(train_bases)} videos en train, {len(val_bases)} en val")
    print(f"  train: {train_bases}")
    print(f"  val:   {val_bases}")

    for split_name, bases in [("train", train_bases), ("val", val_bases)]:
        total_frames = 0
        for base in bases:
            seq_path = pick_variant(all_sequences, base)
            images_out = OUTPUT_DIR / "images" / split_name
            labels_out = OUTPUT_DIR / "labels" / split_name
            n = process_sequence(seq_path, images_out, labels_out)
            total_frames += n
            print(f"  [{split_name}] {seq_path.name}: {n} frames")
        print(f"{split_name}: {total_frames} frames au total")

    write_dataset_yaml(OUTPUT_DIR)
    print(f"\nDataset ecrit dans {OUTPUT_DIR}")
    print(f"Config ultralytics : {OUTPUT_DIR / 'dataset.yaml'}")


def build_pool_dataset() -> None:
    """Mode pool : convertit les 7 videos sans split, + manifeste video/frame.

    Sert de base commune pour la validation croisee -- on convertit et on
    uploade les donnees une seule fois, puis chaque fold repartitionne les
    memes fichiers localement (voir src/detection/kfold.py).
    """
    all_sequences = sorted(MOT17_DIR.glob("train/*"))
    base_names = sorted({base_sequence_name(s.name) for s in all_sequences})

    images_out = POOL_DIR / "images"
    labels_out = POOL_DIR / "labels"
    manifest_rows: list[tuple[str, str]] = []

    total_frames = 0
    for base in base_names:
        seq_path = pick_variant(all_sequences, base)
        n = process_sequence(seq_path, images_out, labels_out)
        for frame_path in sorted(images_out.glob(f"{seq_path.name}_*.jpg")):
            manifest_rows.append((frame_path.name, base))
        total_frames += n
        print(f"  {seq_path.name} -> base {base}: {n} frames")

    POOL_DIR.mkdir(parents=True, exist_ok=True)
    with (POOL_DIR / "manifest.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["filename", "base_sequence"])
        writer.writerows(manifest_rows)

    print(f"\n{total_frames} frames au total sur {len(base_names)} videos")
    print(f"Pool ecrit dans {POOL_DIR}")
    print(f"Manifeste : {POOL_DIR / 'manifest.csv'} ({len(manifest_rows)} lignes)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--mode", choices=["split", "pool"], default="split")
    args = parser.parse_args()

    if args.mode == "split":
        build_split_dataset()
    else:
        build_pool_dataset()


if __name__ == "__main__":
    main()