import cv2
import torch
import torch.nn as nn
import numpy as np
from torchvision import models, transforms
import time

# =====================
# CONFIG
# =====================
MODEL_PATH = "models/gesture_model.pth"
IMAGE_SIZE = 128
NUM_CLASSES = 6
LABELS = ["0","1","2","3","4","5"]
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# =====================
# Load model
# =====================
model = models.mobilenet_v2(pretrained=False)
model.classifier[1] = nn.Linear(model.last_channel, NUM_CLASSES)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model = model.to(DEVICE)
model.eval()

# =====================
# Transform
# =====================
transform = transforms.Compose([
    transforms.ToTensor()
])

# =====================
# Webcam
# =====================
cap = cv2.VideoCapture(0)
prev_time = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Miroir
    frame = cv2.flip(frame, 1)

    h, w = frame.shape[:2]
    size = min(h, w)
    x0 = (w - size)//2
    y0 = (h - size)//2

    crop = frame[y0:y0+size, x0:x0+size]
    img = cv2.resize(crop, (IMAGE_SIZE, IMAGE_SIZE))

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    tensor = transform(img_rgb).unsqueeze(0).to(DEVICE)

    # Inference
    with torch.no_grad():
        outputs = model(tensor)
        probs = torch.softmax(outputs, dim=1)
        conf, pred = torch.max(probs, 1)

    class_id = pred.item()
    confidence = conf.item()

    # FPS
    current_time = time.time()
    fps = 1/(current_time - prev_time) if prev_time != 0 else 0
    prev_time = current_time

    # Display
    cv2.rectangle(frame, (x0,y0), (x0+size,y0+size), (0,255,0), 2)

    cv2.putText(frame, f"Gesture: {LABELS[class_id]}",
                (20,40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

    cv2.putText(frame, f"Conf: {confidence:.2f}",
                (20,80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

    cv2.putText(frame, f"FPS: {int(fps)}",
                (20,120), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

    cv2.imshow("Model Test (PC)", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
