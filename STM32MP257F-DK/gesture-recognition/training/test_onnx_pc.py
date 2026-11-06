import cv2
import numpy as np
import onnxruntime as ort
import time

MODEL_PATH = "models/gesture_model.onnx"
IMAGE_SIZE = 128
LABELS = ["0","1","2","3","4","5"]

session = ort.InferenceSession(MODEL_PATH)
input_name = session.get_inputs()[0].name

cap = cv2.VideoCapture(0)
prev_time = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)

    h, w = frame.shape[:2]
    size = min(h, w)
    x0 = (w - size)//2
    y0 = (h - size)//2

    crop = frame[y0:y0+size, x0:x0+size]
    img = cv2.resize(crop, (IMAGE_SIZE, IMAGE_SIZE))

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32) / 255.0
    img = np.transpose(img, (2,0,1))
    img = np.expand_dims(img, axis=0)

    outputs = session.run(None, {input_name: img})
    preds = outputs[0]

    class_id = np.argmax(preds)
    confidence = np.max(preds)

    current_time = time.time()
    fps = 1/(current_time - prev_time) if prev_time != 0 else 0
    prev_time = current_time

    cv2.putText(frame, f"Gesture: {LABELS[class_id]}", (20,40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

    cv2.putText(frame, f"Conf: {confidence:.2f}", (20,80),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

    cv2.putText(frame, f"FPS: {int(fps)}", (20,120),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

    cv2.imshow("ONNX Test", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
