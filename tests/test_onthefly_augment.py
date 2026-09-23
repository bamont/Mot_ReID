"""Tests du module d'augmentation nuit/flou.
"""

import sys
import types
from unittest.mock import MagicMock

from src.detection.onthefly_augment import (
    enable_domain_augment,
    make_patched_init,
    patch_ultralytics_albumentations,
    supports_native_augmentations,
)


def _install_fake_ultralytics(monkeypatch, albumentations_init) -> types.ModuleType:
    """Construit un faux ultralytics.data.augment.Albumentations avec le
    __init__ donne, et l'enregistre dans sys.modules."""
    fake_augment_module = types.ModuleType("ultralytics.data.augment")

    class FakeAlbumentations:
        __init__ = albumentations_init

    fake_augment_module.Albumentations = FakeAlbumentations

    fake_data_module = types.ModuleType("ultralytics.data")
    fake_data_module.augment = fake_augment_module
    fake_ultralytics_module = types.ModuleType("ultralytics")
    fake_ultralytics_module.data = fake_data_module

    monkeypatch.setitem(sys.modules, "ultralytics", fake_ultralytics_module)
    monkeypatch.setitem(sys.modules, "ultralytics.data", fake_data_module)
    monkeypatch.setitem(sys.modules, "ultralytics.data.augment", fake_augment_module)
    return fake_augment_module


class TestSupportsNativeAugmentations:
    def test_returns_none_if_ultralytics_not_installed(self, monkeypatch):
        monkeypatch.setitem(sys.modules, "ultralytics", None)
        monkeypatch.setitem(sys.modules, "ultralytics.data", None)
        monkeypatch.setitem(sys.modules, "ultralytics.data.augment", None)

        assert supports_native_augmentations() is None

    def test_returns_true_when_transforms_param_present(self, monkeypatch):
        # Simule la signature Ultralytics >= 8.4.15x
        def init_with_transforms(self, p=1.0, transforms=None, flip_idx=None):
            pass

        _install_fake_ultralytics(monkeypatch, init_with_transforms)
        assert supports_native_augmentations() is True

    def test_returns_false_when_transforms_param_absent(self, monkeypatch):
        # Simule l'ancienne signature (celle vue initialement dans les logs)
        def init_without_transforms(self, p=1.0):
            pass

        _install_fake_ultralytics(monkeypatch, init_without_transforms)
        assert supports_native_augmentations() is False


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

    def test_accepts_extra_kwargs_without_crashing(self):
        # C'est exactement ce qui a plante en conditions reelles : Ultralytics
        # 8.4.153 appelle Albumentations(p=1.0, transforms=..., flip_idx=...)
        # -- le patch doit absorber ces kwargs supplementaires sans lever.
        patched_init = make_patched_init(pipeline_factory=lambda: MagicMock())
        instance = types.SimpleNamespace()

        patched_init(instance, p=1.0, transforms=["something"], flip_idx=[0, 1, 2])

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
        def original_init(self, p=1.0):
            self.p = p
            self.transform = "original_hardcoded_pipeline"

        fake_augment_module = _install_fake_ultralytics(monkeypatch, original_init)

        fake_pipeline = MagicMock(name="domain_pipeline")
        result = patch_ultralytics_albumentations(pipeline_factory=lambda: fake_pipeline)
        assert result is True

        instance = fake_augment_module.Albumentations(p=1.0)
        assert instance.transform is fake_pipeline


class TestEnableDomainAugment:
    def test_uses_native_path_when_supported(self, monkeypatch):
        def init_with_transforms(self, p=1.0, transforms=None, flip_idx=None):
            pass

        _install_fake_ultralytics(monkeypatch, init_with_transforms)

        fake_transforms = ["fake_transform_1", "fake_transform_2"]
        monkeypatch.setattr(
            "src.detection.onthefly_augment.build_domain_transforms",
            lambda: fake_transforms,
        )

        train_kwargs = {"epochs": 20, "data": "dataset.yaml"}
        result = enable_domain_augment(train_kwargs)

        assert result["augmentations"] == fake_transforms
        # Les kwargs d'origine ne doivent pas etre perdus
        assert result["epochs"] == 20

    def test_falls_back_to_monkeypatch_when_native_unsupported(self, monkeypatch):
        calls = []

        def init_without_transforms(self, p=1.0):
            pass

        def fake_patch(pipeline_factory=None):
            calls.append("patched")
            return True

        monkeypatch.setattr(
            "src.detection.onthefly_augment.patch_ultralytics_albumentations", fake_patch
        )

        train_kwargs = {"epochs": 20}
        result = enable_domain_augment(train_kwargs)

        assert calls == ["patched"]
        # Pas de clé "augmentations" ajoutée dans ce chemin (le monkey-patch
        # gère tout en interne, pas besoin de modifier train_kwargs)
        assert "augmentations" not in result
        assert result["epochs"] == 20  # kwargs d'origine préservés

    def test_noop_when_ultralytics_not_installed(self, monkeypatch):
        monkeypatch.setitem(sys.modules, "ultralytics", None)
        monkeypatch.setitem(sys.modules, "ultralytics.data", None)
        monkeypatch.setitem(sys.modules, "ultralytics.data.augment", None)

        train_kwargs = {"epochs": 20}
        result = enable_domain_augment(train_kwargs)

        assert result == {"epochs": 20}
