"""Construit le dataset YOLO complet à partir de MOT17.

Usage:
    poetry run python -m src.detection.build_dataset

Produit :
    data/processed/mot17_yolo/
        images/{train,val}/*.jpg   (copies ou liens symboliques des frames)
        labels/{train,val}/*.txt    (annotations YOLO)
        dataset.yaml                 (config pour ultralytics)
"""

from __future__ import annotations

import configparser
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


def process_sequence(seq_path: Path, split: str) -> int:
    """Convertit une sequence complete, ecrit images+labels dans OUTPUT_DIR/split.

    Retourne le nombre de frames traitees.
    """
    img_width, img_height = read_seqinfo(seq_path)
    annotations_by_frame = load_annotations_by_frame(seq_path / "gt" / "gt.txt")

    images_out = OUTPUT_DIR / "images" / split
    labels_out = OUTPUT_DIR / "labels" / split
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


def write_dataset_yaml() -> None:
    content = f"""# Genere par src/detection/build_dataset.py -- ne pas editer a la main
path: {OUTPUT_DIR.resolve()}
train: images/train
val: images/val
names:
  0: pedestrian
"""
    (OUTPUT_DIR / "dataset.yaml").write_text(content)


def main() -> None:
    all_sequences = sorted(MOT17_DIR.glob("train/*"))
    base_names = sorted({base_sequence_name(s.name) for s in all_sequences})
    train_bases, val_bases = split_base_sequences(base_names)

    print(f"Split : {len(train_bases)} videos en train, {len(val_bases)} en val")
    print(f"  train: {train_bases}")
    print(f"  val:   {val_bases}")

    # On ne garde qu'UNE variante (FRCNN par defaut) par video de base --
    # pas besoin des 3 puisque gt.txt est identique entre variantes (verifie en EDA).
    def pick_variant(base_name: str) -> Path:
        candidates = [s for s in all_sequences if s.name.startswith(base_name + "-")]
        preferred = next((s for s in candidates if s.name.endswith("-FRCNN")), None)
        return preferred or candidates[0]

    for split_name, bases in [("train", train_bases), ("val", val_bases)]:
        total_frames = 0
        for base in bases:
            seq_path = pick_variant(base)
            n = process_sequence(seq_path, split_name)
            total_frames += n
            print(f"  [{split_name}] {seq_path.name}: {n} frames")
        print(f"{split_name}: {total_frames} frames au total")

    write_dataset_yaml()
    print(f"\nDataset ecrit dans {OUTPUT_DIR}")
    print(f"Config ultralytics : {OUTPUT_DIR / 'dataset.yaml'}")


if __name__ == "__main__":
    main()
