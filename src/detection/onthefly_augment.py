"""Augmentation a la volée pour la robustesse nuit/flou de mouvement.

Contrainte technique : le pipeline Albumentations d'Ultralytics est construit
en dur dans ultralytics.data.augment.Albumentations.__init__ (Blur, MedianBlur,
ToGray, CLAHE, tous p=0.01). Il n'existe pas de paramètre public pour y injecter 
des transformations personnalisees, donc on remplace cette méthode par monkey-
patch avant d'appeler model.train().
"""

from __future__ import annotations
 
import inspect
from typing import Any
 
 
def build_domain_transforms() -> list[Any]:
    """Liste de transformations Albumentations (pas de Compose) : defauts
    Ultralytics (Blur/MedianBlur/ToGray/CLAHE, p=0.01) + nuit/flou de
    mouvement ajoutes suite au k-fold (MOT17-13 camera embarquee : 0.215,
    MOT17-10 nuit : 0.329, contre 0.60-0.67 sur les scenes de jour a camera
    fixe). Probabilites minoritaires (~20-25%) pour ne
    pas sacrifier les conditions "normales", qui restent la majorite.
    """
    import albumentations as A
 
    return [
        A.Blur(p=0.01),
        A.MedianBlur(p=0.01),
        A.ToGray(p=0.01),
        A.CLAHE(p=0.01),
        A.MotionBlur(blur_limit=(9, 25), p=0.25),
        A.RandomGamma(gamma_limit=(150, 250), p=0.2),
        A.RandomBrightnessContrast(brightness_limit=(-0.6, -0.2), contrast_limit=0.2, p=0.2),
    ]
 
 
def build_domain_pipeline() -> Any:
    """Compose complet (avec bbox_params) -- utilise uniquement par le
    monkey-patch de secours (chemin natif ci-dessous n'en a pas besoin,
    Ultralytics construit son propre Compose autour de la liste retournee
    par build_domain_transforms())."""
    import albumentations as A
 
    return A.Compose(
        build_domain_transforms(),
        bbox_params=A.BboxParams(format="yolo", label_fields=["class_labels"]),
    )
 
 
def supports_native_augmentations() -> bool | None:
    """Detecte si Albumentations.__init__ accepte un parametre 'transforms'.
 
    Retourne True (mecanisme natif dispo), False (version trop ancienne,
    monkey-patch necessaire), ou None (ultralytics pas installe du tout).
    """
    try:
        from ultralytics.data.augment import Albumentations
    except ImportError:
        return None
 
    signature = inspect.signature(Albumentations.__init__)
    return "transforms" in signature.parameters
 
 
def make_patched_init(pipeline_factory=build_domain_pipeline):
    """Cree le __init__ de remplacement pour le monkey-patch de secours.
 
    Accepte **kwargs (pas seulement p) pour rester compatible si la
    signature d'Ultralytics evolue encore -- exactement le piege qui a
    fait planter la premiere version de ce module (elle n'acceptait que
    `p`, la version 8.4.153 appelle aussi `transforms=` et `flip_idx=`).
    Ces arguments additionnels sont ignores : on impose notre pipeline
    complet quoi qu'il arrive dans ce chemin de secours.
    """
 
    def patched_init(self, p: float = 1.0, **_ignored_kwargs: Any) -> None:
        self.p = p
        self.transform = None
        try:
            self.transform = pipeline_factory()
        except ImportError:
            self.transform = None
 
    return patched_init
 
 
def patch_ultralytics_albumentations(pipeline_factory=build_domain_pipeline) -> bool:
    """Remplace Albumentations.__init__ (chemin de secours, versions anciennes).
 
    Retourne True si applique, False si ultralytics n'est pas installe.
    """
    try:
        from ultralytics.data import augment as ultra_augment
    except ImportError:
        return False
 
    ultra_augment.Albumentations.__init__ = make_patched_init(pipeline_factory)
    return True
 
 
def enable_domain_augment(train_kwargs: dict[str, Any]) -> dict[str, Any]:
    """Point d'entree unique : active la robustesse nuit/flou pour ce run.
 
    Modifie et retourne train_kwargs (dict passe a model.train(**kwargs)) :
    - Si le mecanisme natif est disponible, ajoute train_kwargs["augmentations"]
      (Ultralytics l'utilise directement, aucun monkey-patch necessaire).
    - Sinon, si ultralytics est installe mais trop ancien, applique le
      monkey-patch de secours et laisse train_kwargs inchange.
    - Si ultralytics n'est pas installe du tout, ne fait rien (train.py
      plantera de toute facon a l'import de YOLO, plus haut dans le script).
    """
    native = supports_native_augmentations()
    if native:
        train_kwargs["augmentations"] = build_domain_transforms()
    elif native is False:
        patch_ultralytics_albumentations()
    return train_kwargs