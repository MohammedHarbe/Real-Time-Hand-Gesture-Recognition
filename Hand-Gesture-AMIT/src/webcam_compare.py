import cv2
import mediapipe as mp
import numpy as np
from ultralytics import YOLO
from PIL import Image

# ===== EDIT PATHS =====
YOLO_BEST = r"C:\Users\mohamed harbe\runs\classify\runs_amIT\OptionA_YOLOv8n_cls\weights\best.pt"

# Landmark classifier weights
import joblib
LANDMARK_MODEL = joblib.load(r"artifacts/optionB_best_model.joblib")
CLASSES = ["1", "2", "3", "4", "5"]

# ===== YOLO =====
yolo = YOLO(YOLO_BEST)

# ===== Mediapipe =====
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1)

def predict_landmark(frame):
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    res = hands.process(rgb)
    if not res.multi_hand_landmarks:
        return None, 0.0

    lm = res.multi_hand_landmarks[0]
    pts = []
    for p in lm.landmark:
        pts.extend([p.x, p.y])

    pts = np.array(pts).reshape(1, -1)
    pred = LANDMARK_MODEL.predict(pts)[0]
    conf = max(LANDMARK_MODEL.predict_proba(pts)[0])
    return pred, conf

# ===== Webcam =====
cap = cv2.VideoCapture(0)
assert cap.isOpened(), "Webcam not detected"

while True:
    ok, frame = cap.read()
    if not ok:
        break

    # YOLO
    pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    r = yolo.predict(pil, imgsz=224, device=0, verbose=False)[0]
    yolo_idx = int(r.probs.top1)
    yolo_pred = CLASSES[yolo_idx]
    yolo_conf = float(r.probs.top1conf)

    # Landmark
    lm_pred, lm_conf = predict_landmark(frame)

    # Draw
    cv2.putText(frame, f"YOLO: {yolo_pred} ({yolo_conf:.2f})",
                (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)

    if lm_pred is not None:
        cv2.putText(frame, f"Landmark: {lm_pred} ({lm_conf:.2f})",
                    (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,0,0), 2)
    else:
        cv2.putText(frame, "Landmark: No hand",
                    (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)

    cv2.imshow("YOLO vs Landmark Comparison", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()