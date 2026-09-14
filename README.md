# MOT-ReID Pipeline

> Pipeline de **détection**, **suivi multi-objets** et **ré-identification**,
> construit de façon incrémentale, module par module.

[![CI](https://github.com/bamont/Mot_ReID/actions/workflows/ci.yml/badge.svg)](https://github.com/bamont/Mot_ReID/actions)
![Python](https://img.shields.io/badge/python-3.13-blue)

## 🎯 Objectif final

Suivre des individus à travers une vidéo, même après occlusion, en combinant détection,
tracking multi-objets et ré-identification, avec analytics et une démo servie en temps réel.

**Avancement** : setup du repo + exploration des données.

## 📁 Structure du repo

```
.
├── src/
│   └── detection/ 
├── tests/
├── notebooks/
│   └── 01_eda.ipynb   # Exploration des datasets
├── configs/
│   └── datasets.yaml
└── scripts/
    └── download_data.sh
```

## 🚀 Installation

```bash
git clone https://github.com/bamont/mot-reid-pipeline.git
cd mot-reid-pipeline
make install
```

## 🧪 Développement

```bash
make lint     # ruff + black
make format   # auto-fix
make test     # tests unitaires
```

## 📚 Datasets

| Dataset | Usage | Lien |
|---|---|---|
| MOT17 | Détection & tracking | [motchallenge.net](https://motchallenge.net/data/MOT17/) |
| Market-1501 | Ré-identification | [lien projet](https://zheng-lab.cecs.anu.edu.au/Project/project_reid.html) |

Voir `configs/datasets.yaml` et `scripts/download_data.sh`.

## 📜 Licence

MIT
