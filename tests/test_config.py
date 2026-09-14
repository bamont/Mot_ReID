"""Tests de sanite pour la configuration du projet."""

from src.config import CONFIGS_DIR, DATA_PROCESSED, DATA_RAW, PROJECT_ROOT


def test_project_root_exists():
    assert PROJECT_ROOT.exists()
    assert (PROJECT_ROOT / "pyproject.toml").exists()


def test_data_dirs_defined():
    assert DATA_RAW.name == "raw"
    assert DATA_PROCESSED.name == "processed"


def test_configs_dir_exists():
    assert CONFIGS_DIR.exists()
    assert (CONFIGS_DIR / "datasets.yaml").exists()
