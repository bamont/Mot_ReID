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

## Modèle de détection final retenu : YOLO11n brut (zéro fine-tuning)

**Décision** : le détecteur utilisé dans le pipeline final de trakcing est
**YOLO11n pré-entraîné COCO, sans fine-tuning**, filtré à la classe "person" (`classes=[0]`).

**Alternatives considérées** :
- YOLO11n fine-tuné sur MOT17 (`freeze=10`, sans augmentation) : mAP50-95 = 0.465 ± 0.158
- YOLO11n brut (zero-shot, aucun entraînement) : mAP50-95 = 0.458 ± 0.139

**Raison** : les deux options sont quasi équivalentes en moyenne (écart de 0.007), 
mais leur comportement diffère par vidéo de façon interprétable : le fine-tuning
gagne sur les scènes "typiques" MOT17 (jour, caméra fixe — ex: MOT17-09 +0.048) et perd sur les
scènes atypiques (caméra mobile, ex: MOT17-13 -0.032). Le zero-shot a un écart-type plus faible
(0.139 vs 0.158) : plus stable d'une scène à l'autre. Le zero-shot est le choix le plus sûr et 
évite en plus toute la complexité (pipeline d'entraînement, gestion des poids, risque de 
sur-apprentissage aux 6 scènes d'entraînement) pour un bénéfice non évident.