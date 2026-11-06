import os
import numpy as np
import cv2

DATASET_PATH = "dataset"
IMAGE_SIZE = 128

samples = []

for class_name in os.listdir(DATASET_PATH):
    class_path = os.path.join(DATASET_PATH, class_name)
    for img_name in os.listdir(class_path)[:50]:  # 50 images par classe
        img_path = os.path.join(class_path, img_name)
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE))
        img = img.astype(np.float32) / 255.0
        img = np.transpose(img, (2,0,1))
        samples.append(img)

samples = np.array(samples)
np.savez("calibration.npz", input=samples)

print("Calibration file created.")
