# Journal des décisions techniques

Ce fichier trace les choix d'architecture et leurs justifications au fil du projet.
Format : date, décision, alternatives considérées, raison du choix.

---

## 2026-XX-XX — Croissance incrémentale du repo

**Décision** : ne créer un dossier/module (`tracking/`, `reid/`, `api/`...) et n'ajouter une
dépendance (fastapi, wandb, onnx, filterpy...) que lorsque le chantier correspondant démarre
réellement, plutôt que de tout scaffolder à l'avance.

**Alternatives considérées** : générer toute l'arborescence et toutes les dépendances dès la
semaine 1, en anticipant les besoins des semaines suivantes.

**Raison** : un repo avec des dossiers vides et des dépendances lourdes (torch, fastapi,
postgres...) installées sans code qui les utilise est trompeur pour qui consulte le projet,
augmente la surface de maintenance (CI plus lente, plus de choses à mettre à jour) et complique
le diagnostic si quelque chose casse. Chaque semaine du plan ajoute exactement ce dont elle a
besoin, documenté ici au moment de l'ajout.

---

## 2026-XX-XX — Choix des datasets

**Décision** : MOT17 pour la détection/tracking, Market-1501 pour la ré-identification.

**Alternatives considérées** :
- MOT20 (scènes plus denses, plus difficile) — gardé en option pour une évaluation de robustesse plus tard
- DukeMTMC (retiré de la distribution publique pour raisons éthiques) — écarté
- Dataset vidéo personnel — envisagé pour une démo plus originale, à ajouter en semaine 6-7

**Raison** : MOT17 est le standard de facto pour benchmarker un tracker, ce qui permet de comparer
mes résultats à la littérature. Market-1501 est similairement la référence pour la ré-ID.

---

## 2026-XX-XX — Gestionnaire de dépendances

**Décision** : Poetry plutôt que pip/requirements.txt ou conda.

**Raison** : lock file reproductible, gestion propre des groupes de dépendances (dev vs prod),
meilleure intégration avec les outils de packaging.

---

## Template pour les prochaines entrées

```
## YYYY-MM-DD — Titre de la décision

**Décision** : ...

**Alternatives considérées** :
- ...

**Raison** : ...
```
