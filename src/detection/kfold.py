"""Prépare un fold de validation croisée

Principe : les 7 videos MOT17 sont converties UNE SEULE FOIS dans un pool
(images/*.jpg, labels/*.txt, manifest.csv), puis chaque fold repartitionne
les memes fichiers en train/val par lien symbolique.
Un fold = une video de base tenue a l'ecart comme validation, les 6 autres
servant a l'entrainement.
"""

from __future__ import annotations
 
import csv
import os
import shutil
from pathlib import Path
 
 
def read_manifest(manifest_path: Path) -> list[tuple[str, str]]:
    """Lit manifest.csv -> liste de (filename, base_sequence)."""
    with manifest_path.open(newline="") as f:
        reader = csv.DictReader(f)
        return [(row["filename"], row["base_sequence"]) for row in reader]
 
 
def list_base_sequences(manifest_rows: list[tuple[str, str]]) -> list[str]:
    """Liste triee des videos de base presentes dans le manifeste."""
    return sorted({base for _, base in manifest_rows})
 
 
def assign_split(manifest_rows: list[tuple[str, str]], held_out_base: str) -> dict[str, str]:
    """Attribue chaque fichier a 'train' ou 'val' pour ce fold.
 
    held_out_base est la SEULE video en validation ; toutes les autres vont
    en train. Retourne {filename: "train"|"val"}.
    """
    return {
        filename: ("val" if base == held_out_base else "train")
        for filename, base in manifest_rows
    }
 
 
def _link_or_copy(src: Path, dst: Path) -> None:
    """Cree dst pointant sur le contenu de src : symlink, sinon hard link, sinon copie.
 
    Essaie dans cet ordre le moins couteux au plus couteux en espace disque.
    Le symlink echoue silencieusement sans privilege sous Windows (OSError),
    d'ou le repli automatique sur le hard link, puis la copie en dernier
    recours (ex: systemes de fichiers qui ne supportent ni l'un ni l'autre).
    """
    try:
        dst.symlink_to(src)
        return
    except OSError:
        pass
    try:
        os.link(src, dst)
        return
    except OSError:
        pass
    shutil.copy2(src, dst)
 
 
def prepare_fold(
    pool_dir: Path,
    manifest_rows: list[tuple[str, str]],
    held_out_base: str,
    fold_output_dir: Path,
) -> tuple[int, int]:
    """Cree fold_output_dir/{images,labels}/{train,val} a partir du pool.
 
    Retourne (n_train, n_val).
    """
    assignment = assign_split(manifest_rows, held_out_base)
 
    for split_name in ("train", "val"):
        (fold_output_dir / "images" / split_name).mkdir(parents=True, exist_ok=True)
        (fold_output_dir / "labels" / split_name).mkdir(parents=True, exist_ok=True)
 
    n_train = n_val = 0
    for filename, split_name in assignment.items():
        label_name = filename.rsplit(".", 1)[0] + ".txt"
 
        img_dst = fold_output_dir / "images" / split_name / filename
        label_dst = fold_output_dir / "labels" / split_name / label_name
 
        img_src = pool_dir / "images" / filename
        label_src = pool_dir / "labels" / label_name
 
        if not img_dst.exists():
            _link_or_copy(img_src.resolve(), img_dst)
        if not label_dst.exists():
            _link_or_copy(label_src.resolve(), label_dst)
 
        if split_name == "train":
            n_train += 1
        else:
            n_val += 1
 
    write_fold_dataset_yaml(fold_output_dir)
    return n_train, n_val
 
 
def write_fold_dataset_yaml(fold_output_dir: Path) -> None:
    content = f"""# Genere par src/detection/kfold.py -- ne pas editer a la main
path: {fold_output_dir.resolve()}
train: images/train
val: images/val
names:
  0: pedestrian
"""
    (fold_output_dir / "dataset.yaml").write_text(content)