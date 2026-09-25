## Choix des datasets

**Décision** : MOT17 pour la détection/tracking, Market-1501 pour la ré-identification.

**Raison** : MOT17 est le standard pour benchmarker un tracker. Market-1501 est similairement la référence pour la ré-ID.

**Alternatives considérées** :
- MOT20 : scènes plus denses, plus difficile. Gardé en option pour une évaluation de robustesse plus tard

## Validation croisée 7-fold : baseline de détection

**Décision** : la baseline de détection retenue pour le rapport est
**yolo11n, freeze=10, sans domain-augment : mAP50-95 = 0.465 ± 0.158** (moyenne ± écart-type
sur validation croisée 7-fold, chaque vidéo MOT17 tenue à l'écart à tour de rôle).

**Alternatives testées** :
1. `freeze=10` sans augmentation ciblée (baseline) → 0.465 ± 0.158
2. `freeze=10` + augmentation nuit/flou de mouvement à la volée (MotionBlur, RandomGamma,
   RandomBrightnessContrast via le paramètre natif `augmentations=` d'Ultralytics) → 0.461 ± 0.154
   (aucune amélioration significative, dans le bruit)
3. `freeze=0` (backbone dégelé) + même augmentation → 0.423 ± 0.171, avec un effondrement sévère
   sur MOT17-13 (0.120, le pire résultat toutes configs confondues, pic à l'epoch 1 puis
   dégradation continue)

**Raison de l'abandon de la piste "robustesse nuit/mouvement"** : la validation croisée avait
révélé une perte de performance marquée sur 2 vidéos aux conditions visuelles spécifiques
(MOT17-13, caméra embarquée en mouvement : 0.215 ; MOT17-10, scène de nuit : 0.329) contre
0.60-0.67 sur les 5 vidéos de jour à caméra fixe. Deux hypothèses ont été testées pour corriger
cela : 
(a) exposer le modèle à des variantes synthétiques nuit/flou pendant l'entraînement
(b) dégeler le backbone pour laisser les features bas-niveau s'adapter à ces conditions. 
Aucune des deux n'a amélioré la situation. La seconde l'a même dégradée, en réintroduisant
le sur-apprentissage des 6 scènes d'entraînement qu'on avait justement cherché à éviter avec le
gel du backbone.

**Conclusion retenue** : le facteur limitant n'est pas résoluble par réglage
d'hyperparamètres ou augmentation de données synthétiques. C'est un problème de
diversité du jeu d'entraînement. Une vraie solution demanderait des données d'entraînement
supplémentaires couvrant les conditions particulières évoquées. 

## Ajout de RT-DETR à la comparaison : confirmation croisée du phénomène fine-tuning/zero-shot

**Contexte** : suite à la décision precedente (YOLO11n zero-shot), un modèle d'architecture différente (RT-DETR-l, CNN+transformer, ~32M parametres vs ~2.6M pour YOLO11n) a été ajouté à la comparaison, en zero-shot et fine-tune (freeze=10, même protocole 7-fold).

**Résultats** (mAP50-95, moyenne +/- écart-type sur les 7 folds) :

| Configuration | Moyenne | Ecart-type |
|---|---|---|
| YOLO11n zero-shot | 0.458 | 0.139 |
| YOLO11n fine-tune (freeze=10) | 0.465 | 0.158 |
| RT-DETR fine-tune (freeze=10) | 0.509 | 0.157 |
| **RT-DETR zero-shot** | **0.511** | **0.119** |

**Confirmation croisée importante** : le phénomène observé avec YOLO11n (le fine-tuning améliore les scènes typiques et dégrade les scenes atypiques) se reproduit à l'identique avec RT-DETR.

**Décision** : le modèle de détection final retenu est RT-DETR-l pré-entrainé COCO, sans fine-tuning (zero-shot), filtré a la classe "person" (classes=[0]). Remplace la decision precedente (YOLO11n zero-shot).

**Raison** : RT-DETR zero-shot domine strictement YOLO11n zero-shot sur les deux métriques qui comptent (mAP50-95 plus élevé : 0.511 vs 0.458 ; plus stable : écart-type 0.119 vs 0.139), sans aucun coût d'entrainement supplémentaire.

**Compromis assumé** : RT-DETR est ~13x plus lent en inférence (~37ms/image sur T4, soit ~27 FPS max, contre ~3ms/~300 FPS pour YOLO11n). Choix justifié ici car la précision/stabilité prime sur le débit pour ce projet. 27 FPS reste largement suffisant pour une démo vidéo (au-dela du seuil de fluidité vidéo standard à 25 FPS), et le pipeline n'a pas de contrainte de flux temps réel strict.

## Construction du dataset de triplets pour la ré-ID (Market-1501)

**Décision** : split train/val **par identité** (et non par image) : les identités de
validation sont totalement absentes du pool d'entraînement.

**Raison** : Market-1501 fournit déjà 751 identités de train disjointes des 750 identités 
de test. Un split par image aurait laissé la même personne apparaître en train et
 en val : le modèle reconnaîtrait l'identité plutôt que d'apprendre
une représentation généralisable, et le mAP de validation serait optimiste.

**Décision** : positif échantillonné **en priorité sur une autre caméra** que l'ancre
(fallback même-caméra si l'identité n'a qu'une seule caméra disponible).

**Raison** : un positif pris sur la même caméra que l'ancre (souvent quelques frames plus
loin, même pose/éclairage/arrière-plan) est une tâche trop facile : le réseau peut
s'appuyer sur des indices de bas niveau plutôt que sur l'identité de la personne. Un
positif cross-caméra force une vraie invariance de point de vue, et correspond exactement
au protocole d'évaluation de Market-1501 (la query et la galerie proviennent toujours de
caméras différentes).

**Décision** : négatif tiré **aléatoirement** parmi les autres identités (pas de hard
negative mining).

**Raison** : le hard negative mining (choisir le négatif le plus proche de l'ancre dans
l'espace d'embedding courant) nécessite un forward pass du modèle en cours d'entraînement : 
ça relève de la boucle d'entraînement elle-même, pas de la construction du dataset.
Piste à explorer plus tard au niveau de `train.py`.

**Décision** : échantillonnage **à la volée** (`TripletMarket1501Dataset`, un triplet frais
par `__getitem__`) plutôt qu'une liste de triplets figée une fois pour toutes.

**Raison** : une liste pré-générée limite l'entraînement à un nombre fixe de combinaisons
ancre/positif/négatif : sur plusieurs dizaines d'époques le modèle finit par revoir
toujours les mêmes triplets. L'échantillonnage à la volée (via `set_epoch`, même convention
que `DistributedSampler`) garde la reproductibilité (seed fixe -> mêmes triplets à époque
donnée) tout en variant les combinaisons vues au fil de l'entraînement. `build_dataset.py`
exporte quand même un `sample_triplets.csv` figé (5000 triplets par défaut) pour
inspection manuelle et illustration dans le rapport, mais ce n'est pas ce qui sert à
l'entraînement.

## Du négatif aleatoire au batch-hard mining (constat empirique)

**Constat** : premier entraînement (négatif aléatoire, `TripletMarket1501Dataset`, 30
époques) : `train_loss` s'effondre dès l'époque 3 (0.089 → 0.003), `val_margin_ok` monte
à ~93%. Mais le vrai mAP mesuré sur `query/`+`bounding_box_test/` (protocole officiel,
`src/reid/evaluate.py`) n'est que de **24.9%** (rank-1 42.6%), très en dessous des
baselines triplet loss publiées sur Market-1501 (~65-75% mAP).

**Diagnostic** : `val_margin_ok` ne teste jamais le modèle contre un négatif difficile --
avec un négatif tiré au hasard parmi 750 autres identités, la quasi-totalité des triplets
sont triviaux dès que l'espace d'embedding est grossièrement organisé. Le modèle "réussit"
sa métrique de suivi sans plus apprendre, alors qu'un vrai retrieval sur 13 102 images de
galerie confronte constamment le modèle à des négatifs proches (autre personne portant des
vêtements similaires, même pose, etc.).

**Décision** : passer au **batch-hard mining** (Hermans et al. 2017) : batches de P
identités x K images, positif/négatif les plus durs minés à l'intérieur de chaque batch
plutôt qu'un triplet aléatoire pré-échantillonné. Implémenté dans `src/reid/pk_sampling.py`
+ `--mining batch-hard` (nouveau défaut de `train.py`), tout en gardant `--mining random`
disponible pour comparaison directe des deux courbes de mAP sur les mêmes données.

**Résultat** : même protocole d'évaluation (`src/reid/evaluate.py`), mêmes 676 identités
train / 75 val, 30 époques dans les deux cas :

| Mining | val_margin_ok (fin d'entraînement) | mAP réel | Rank-1 |
|---|---|---|---|
| random (négatif aléatoire) | 93.0% | 24.9% | 42.6% |
| batch-hard (P=16, K=4) | 46.5% | **63.8%** | **80.8%** |

Confirme le diagnostic : `val_margin_ok` est nettement PLUS BAS en batch-hard (46% vs 93%)
alors que le vrai mAP est bien MEILLEUR (×2.5) -- les deux métriques ne mesurent pas la
même difficulté, val_margin_ok en batch-hard reste un indicateur utile (sa progression
27%→49% sur l'entraînement montre que le modèle continue d'apprendre) mais les valeurs
absolues ne sont pas comparables entre les deux modes. mAP obtenu (63.8%) proche des
baselines triplet loss publiées sur Market-1501 (~65-75%), sans encore de re-ranking ni de
scheduler de learning rate -- `val_loss` plafonne des l'epoque ~10 (0.12-0.14) pendant que
`train_loss` continue de baisser (0.29→0.009) : signe de sur-apprentissage naissant,
piste suivante pour aller chercher les derniers points.
