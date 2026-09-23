"""Configuration centralisee du projet (chemins, constantes)."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_RAW = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
CONFIGS_DIR = PROJECT_ROOT / "configs"

MOT17_DIR = DATA_RAW / "MOT17"
MARKET1501_DIR = DATA_RAW / "Market-1501"
MARKET1501_TRAIN_DIR = MARKET1501_DIR / "bounding_box_train"
MARKET1501_TEST_DIR = MARKET1501_DIR / "bounding_box_test"
MARKET1501_QUERY_DIR = MARKET1501_DIR / "query"

REID_TRIPLETS_DIR = DATA_PROCESSED / "market1501_triplets"
