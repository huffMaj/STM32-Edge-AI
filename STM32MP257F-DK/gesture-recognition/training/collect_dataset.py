import cv2
import os
import time

# =====================
# CONFIG
# =====================
DATASET_PATH = "dataset"
IMAGE_SIZE = 128
CLASSES = ["0","1","2","3","4","5"]

auto_mode = False
auto_fps = 5
last_capture_time = 0

# =====================
# Setup dossiers
# =====================
for c in CLASSES:
    os.makedirs(os.path.join(DATASET_PATH, c), exist_ok=True)

def get_next_index(class_name):
    files = os.listdir(os.path.join(DATASET_PATH, class_name))
    ids = [int(f.split(".")[0]) for f in files if f.split(".")[0].isdigit()]
    return max(ids)+1 if ids else 0

cap = cv2.VideoCapture(0)

current_class = "0"
counters = {c: get_next_index(c) for c in CLASSES}

print("0-5=label | S=save | A=auto | +/-=speed | Q=quit")

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
    crop_resized = cv2.resize(crop, (IMAGE_SIZE, IMAGE_SIZE))

    # Rectangle guide
    cv2.rectangle(frame, (x0,y0), (x0+size,y0+size), (0,255,0), 2)

    # Overlay
    cv2.putText(frame, f"Class: {current_class}", (20,40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

    cv2.putText(frame, f"Count: {counters[current_class]}", (20,80),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

    cv2.putText(frame, f"Auto: {'ON' if auto_mode else 'OFF'} ({auto_fps} img/s)", 
                (20,120), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)

    cv2.imshow("Dataset Collector", frame)

    # =====================
    # AUTO CAPTURE
    # =====================
    if auto_mode:
        now = time.time()
        if now - last_capture_time > 1/auto_fps:
            save_path = os.path.join(DATASET_PATH, current_class,
                                     f"{counters[current_class]}.jpg")
            cv2.imwrite(save_path, crop_resized)
            counters[current_class] += 1
            last_capture_time = now

    # =====================
    # KEYBOARD
    # =====================
    key = cv2.waitKey(1) & 0xFF

    if key == ord('q'):
        break

    if key in [ord(str(i)) for i in range(6)]:
        current_class = chr(key)

    if key == ord('s'):
        save_path = os.path.join(DATASET_PATH, current_class,
                                 f"{counters[current_class]}.jpg")
        cv2.imwrite(save_path, crop_resized)
        counters[current_class] += 1

    if key == ord('a'):
        auto_mode = not auto_mode

    if key == ord('+') or key == ord('='):
        auto_fps = min(20, auto_fps + 1)

    if key == ord('-'):
        auto_fps = max(1, auto_fps - 1)

cap.release()
cv2.destroyAllWindows()
