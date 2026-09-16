"""Augmentation a la volée pour la robustesse nuit/flou de mouvement.

Contrainte technique : le pipeline Albumentations d'Ultralytics est construit
en dur dans ultralytics.data.augment.Albumentations.__init__ (Blur, MedianBlur,
ToGray, CLAHE, tous p=0.01). Il n'existe pas de paramètre public pour y injecter 
des transformations personnalisees, donc on remplace cette méthode par monkey-
patch avant d'appeler model.train().
"""

from __future__ import annotations

from typing import Any


def build_domain_pipeline() -> Any:
    """Construit le pipeline Albumentations : defauts Ultralytics + nuit/flou.

    Garde les transformations d'origine et ajoute :
    - MotionBlur : simule une camera embarquée en mouvement (cf. MOT17-13)
    - RandomGamma / RandomBrightnessContrast assombrissants : simulent une
      scene de nuit (cf. MOT17-10)
    """
    import albumentations as A

    transforms = [
        A.Blur(p=0.01),
        A.MedianBlur(p=0.01),
        A.ToGray(p=0.01),
        A.CLAHE(p=0.01),
        A.MotionBlur(blur_limit=(9, 25), p=0.25),
        A.RandomGamma(gamma_limit=(150, 250), p=0.2),
        A.RandomBrightnessContrast(brightness_limit=(-0.6, -0.2), contrast_limit=0.2, p=0.2),
    ]
    return A.Compose(
        transforms,
        bbox_params=A.BboxParams(format="yolo", label_fields=["class_labels"]),
    )


def make_patched_init(pipeline_factory=build_domain_pipeline):
    """Cree la fonction __init__ de remplacement pour Albumentations.

    Isolee dans sa propre fonction (plutot qu'inline dans patch_*) pour etre
    testable independamment de l'import reel d'ultralytics/albumentations.
    """

    def patched_init(self, p: float = 1.0) -> None:
        self.p = p
        self.transform = None
        try:
            self.transform = pipeline_factory()
        except ImportError:
            self.transform = None

    return patched_init


def patch_ultralytics_albumentations(pipeline_factory=build_domain_pipeline) -> bool:
    """Remplace Albumentations.__init__ dans ultralytics.data.augment.

    A appeler AVANT model.train(). Retourne True si le patch a ete applique,
    False si ultralytics n'est pas installe (permet un appel sans risque
    depuis un environnement de test).
    """
    try:
        from ultralytics.data import augment as ultra_augment
    except ImportError:
        return False

    ultra_augment.Albumentations.__init__ = make_patched_init(pipeline_factory)
    return True