"""Tests du monkey-patch d'augmentation a la volée.
"""

import sys
import types
from unittest.mock import MagicMock

import pytest

from src.detection.onthefly_augment import (
    make_patched_init,
    patch_ultralytics_albumentations,
)


class TestMakePatchedInit:
    def test_sets_p_attribute(self):
        fake_pipeline = MagicMock()
        patched_init = make_patched_init(pipeline_factory=lambda: fake_pipeline)

        instance = types.SimpleNamespace()
        patched_init(instance, p=0.9)

        assert instance.p == 0.9

    def test_sets_transform_from_factory(self):
        fake_pipeline = MagicMock(name="fake_compose")
        patched_init = make_patched_init(pipeline_factory=lambda: fake_pipeline)

        instance = types.SimpleNamespace()
        patched_init(instance, p=1.0)

        assert instance.transform is fake_pipeline

    def test_default_p_is_one(self):
        patched_init = make_patched_init(pipeline_factory=lambda: MagicMock())
        instance = types.SimpleNamespace()
        patched_init(instance)
        assert instance.p == 1.0

    def test_falls_back_gracefully_if_factory_raises_import_error(self):
        def failing_factory():
            raise ImportError("albumentations not installed")

        patched_init = make_patched_init(pipeline_factory=failing_factory)
        instance = types.SimpleNamespace()
        patched_init(instance, p=1.0)

        assert instance.transform is None


class TestPatchUltralyticsAlbumentations:
    def test_returns_false_if_ultralytics_not_installed(self, monkeypatch):
        monkeypatch.setitem(sys.modules, "ultralytics", None)
        monkeypatch.setitem(sys.modules, "ultralytics.data", None)
        monkeypatch.setitem(sys.modules, "ultralytics.data.augment", None)

        result = patch_ultralytics_albumentations(pipeline_factory=lambda: MagicMock())
        assert result is False

    def test_replaces_init_on_fake_ultralytics_module(self, monkeypatch):
        fake_augment_module = types.ModuleType("ultralytics.data.augment")

        class FakeAlbumentations:
            def __init__(self, p: float = 1.0) -> None:
                self.p = p
                self.transform = "original_hardcoded_pipeline"

        fake_augment_module.Albumentations = FakeAlbumentations

        fake_data_module = types.ModuleType("ultralytics.data")
        fake_data_module.augment = fake_augment_module
        fake_ultralytics_module = types.ModuleType("ultralytics")
        fake_ultralytics_module.data = fake_data_module

        monkeypatch.setitem(sys.modules, "ultralytics", fake_ultralytics_module)
        monkeypatch.setitem(sys.modules, "ultralytics.data", fake_data_module)
        monkeypatch.setitem(sys.modules, "ultralytics.data.augment", fake_augment_module)

        fake_pipeline = MagicMock(name="domain_pipeline")
        result = patch_ultralytics_albumentations(pipeline_factory=lambda: fake_pipeline)
        assert result is True

        instance = fake_augment_module.Albumentations(p=1.0)
        assert instance.transform is fake_pipeline
        assert instance.transform != "original_hardcoded_pipeline"


class TestBuildDomainPipelineContract:
    """Sans importer le vrai albumentations (lourd/instable dans ce sandbox),
    on documente au moins que build_domain_pipeline leve ImportError proprement
    si albumentations est absent -- comportement attendu par make_patched_init.
    """

    def test_raises_import_error_when_albumentations_missing(self, monkeypatch):
        import builtins

        real_import = builtins.__import__

        def fake_import(name, *args, **kwargs):
            if name == "albumentations":
                raise ImportError("no albumentations")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", fake_import)

        from src.detection.onthefly_augment import build_domain_pipeline

        with pytest.raises(ImportError):
            build_domain_pipeline()