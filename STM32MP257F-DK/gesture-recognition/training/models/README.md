# Modèles

- `gesture_model.pth` : poids PyTorch du modèle MobileNetV2 entraîné (6 classes).
- `gesture_model.onnx` : export ONNX de ce modèle, avant quantification.

La version quantifiée utilisée sur la carte est `gesture_model_PerChannel_quant.onnx`, dans le dossier `gesture-recognition`.
