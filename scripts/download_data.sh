#!/usr/bin/env bash
# Telechargement des datasets du projet.
# Usage: bash scripts/download_data.sh [mot17|market1501|all]

set -euo pipefail

RAW_DIR="data/raw"
mkdir -p "$RAW_DIR"

download_mot17() {
    echo ">> Telechargement de MOT17..."
    echo "MOT17 necessite une inscription sur https://motchallenge.net/"
    echo "Telecharge manuellement 'MOT17.zip' puis place-le dans ${RAW_DIR}/"
    echo "Ensuite : unzip MOT17.zip -d ${RAW_DIR}/MOT17"
}

download_market1501() {
    echo ">> Telechargement de Market-1501..."
    echo "Dataset disponible sur Kaggle : "
    echo "  kaggle datasets download -d pengcw1/market-1501"
    echo "ou via le lien officiel du labo ANU (voir configs/datasets.yaml)."
    echo "Place l'archive extraite dans ${RAW_DIR}/Market-1501"
}

case "${1:-all}" in
    mot17) download_mot17 ;;
    market1501) download_market1501 ;;
    all)
        download_mot17
        download_market1501
        ;;
    *)
        echo "Usage: $0 [mot17|market1501|all]"
        exit 1
        ;;
esac

echo ""
echo "Rappel : ces datasets necessitent souvent une inscription/acceptation"
echo "de licence, d'ou l'absence de telechargement direct automatise."
