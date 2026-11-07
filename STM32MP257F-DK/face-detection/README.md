# Détection de visages BlazeFace sur STM32MP257F-DK

Partie « détection de visages » du projet, côté microprocesseur (OpenSTLinux).

## Application sur la carte : `face_detection.py`

- Capture caméra avec GStreamer (`libcamerasrc`, 640 × 480).
- Redimensionnement de l'image en 128 × 128 et normalisation entre 0 et 1.
- Inférence du modèle quantifié `BlazeFace_Optimized.onnx` avec ONNX Runtime, sur le NPU (`VSINPUExecutionProvider`), avec repli sur le CPU pour les opérations non prises en charge.
- Décodage des 896 ancres, seuil de confiance à 0,7, suppression des non-maxima (IoU 0,3, dix visages au maximum).
- Affichage GTK : boîtes englobantes, nombre de visages, images par seconde (inférence et affichage).

Prérequis sur la carte : Python 3, ONNX Runtime avec le fournisseur VSINPU, GStreamer avec `libcamerasrc`, PyGObject (GTK 3) et pycairo.

```bash
python3 face_detection.py
```

Résultat mesuré pendant le projet : 18 images par seconde sur la carte, pour un temps d'inférence estimé à 25 ms par ST Edge AI Developer Cloud.

## Préparation du modèle : `BlazeFace-PyTorch/`

Implémentation PyTorch de BlazeFace, un détecteur de visages rapide et léger. Ce dossier permet de :

- charger un modèle pré-entraîné (`blazeface.pth` ou `blazefaceback.pth`) ;
- détecter des visages et six points clés faciaux ;
- exporter le modèle en ONNX (`export_onnx.py`) ;
- générer un jeu de calibration NPZ pour la quantification INT8 (`make_calib_npz.py`).

Le modèle frontal est conçu pour des visages assez proches de la caméra (cas selfie), pas pour de très petits visages.

### Installation

Python 3.8 ou plus récent :

```bash
pip install torch numpy opencv-python notebook
```

### Prise en main

```bash
cd STM32MP257F-DK/face-detection/BlazeFace-PyTorch
jupyter notebook Inference.ipynb   # démonstration d'inférence
python export_onnx.py              # export ONNX -> blazeface.onnx
python make_calib_npz.py           # jeu de calibration -> blazeface_calib.npz
```

Le notebook `Inference.ipynb` attend des photos de test (`1face.png`, `3faces.png`, `4faces.png`) et `make_calib_npz.py` lit les images du dossier `calib_images/`. Ces images ne sont pas fournies : ajoutez les vôtres.

### Fichiers

| Fichier | Rôle |
|---|---|
| `blazeface.py` | Modèle BlazeFace et post-traitement (décodage, NMS) |
| `blazeface.pth`, `blazefaceback.pth` | Poids des modèles caméra frontale et caméra arrière |
| `anchors.npy`, `anchorsback.npy` | Ancres des deux modèles |
| `blazeface.onnx` | Export ONNX du modèle frontal |
| `export_onnx.py` | Export du modèle PyTorch vers ONNX |
| `make_calib_npz.py` | Préparation du jeu de calibration au format NPZ |
| `Inference.ipynb` | Démonstration d'inférence |
| `Convert.ipynb` | Conversion des poids depuis la version TFLite |
| `Anchors.ipynb` | Génération des ancres |

## Ressources

- BlazeFace (Google Research) : https://sites.google.com/view/perception-cv4arvr/blazeface
- Article : https://arxiv.org/abs/1907.05047
- MediaPipe Face Detection : https://github.com/google/mediapipe
