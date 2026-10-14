"""
Real-time sign language (ASL alphabet) recognition from a webcam.

Pipeline
--------
webcam frame -> MediaPipe HandLandmarker (21 hand points)
             -> normalise the points (same as in training)
             -> trained Keras neural network -> predicted sign + confidence
             -> hold a sign steadily to type it; letters build up into words

Run:
    python realtime_sign_recognition.py
    python realtime_sign_recognition.py --flip-input     # mirror the camera image

Keys (click the video window first):
    q = quit      c = clear the text      b = delete the last character
"""
import argparse
import json
import os
import sys
import time
import urllib.request
from collections import Counter, deque

import cv2
import mediapipe as mp
import numpy as np
from tensorflow import keras

HAND_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/latest/hand_landmarker.task"
)
HAND_MODEL_FILE = "hand_landmarker.task"

# Which of the 21 points are joined by a line (standard MediaPipe hand skeleton).
CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),          # thumb
    (0, 5), (5, 6), (6, 7), (7, 8),          # index finger
    (5, 9), (9, 10), (10, 11), (11, 12),     # middle finger
    (9, 13), (13, 14), (14, 15), (15, 16),   # ring finger
    (13, 17), (17, 18), (18, 19), (19, 20),  # little finger
    (0, 17),                                 # palm edge
]


def ensure_hand_model():
    """Download the MediaPipe hand model file the first time the script runs."""
    if not os.path.exists(HAND_MODEL_FILE):
        print("Downloading hand_landmarker.task ...")
        urllib.request.urlretrieve(HAND_MODEL_URL, HAND_MODEL_FILE)


def landmarks_to_features(hand):
    """21 landmarks -> 63 numbers. MUST match the preprocessing used in training."""
    pts = np.array([[p.x, p.y, p.z] for p in hand], dtype=np.float32)
    pts = pts - pts[0]                 # position relative to the wrist
    scale = np.max(np.abs(pts))
    if scale == 0:
        return None
    pts = pts / scale                  # remove the effect of hand size / distance
    return pts.flatten()


def draw_hand(frame, hand):
    h, w = frame.shape[:2]
    xy = [(int(p.x * w), int(p.y * h)) for p in hand]
    for a, b in CONNECTIONS:
        cv2.line(frame, xy[a], xy[b], (255, 200, 0), 2)
    for x, y in xy:
        cv2.circle(frame, (x, y), 4, (0, 255, 0), -1)


def main():
    parser = argparse.ArgumentParser(description="Real-time sign language recognition")
    parser.add_argument("--camera", type=int, default=0, help="webcam index (default 0)")
    parser.add_argument("--model", default="sign_model.keras", help="trained Keras model")
    parser.add_argument("--classes", default="classes.json", help="JSON list of class names")
    parser.add_argument("--flip-input", action="store_true",
                        help="mirror the camera image before detection (try this if "
                             "predictions are consistently wrong)")
    parser.add_argument("--hold", type=float, default=1.2,
                        help="seconds a sign must be held steady to be typed")
    parser.add_argument("--min-conf", type=float, default=0.80,
                        help="minimum confidence to accept a prediction")
    parser.add_argument("--smooth", type=int, default=8,
                        help="number of recent frames used for majority voting")
    args = parser.parse_args()

    for path in (args.model, args.classes):
        if not os.path.exists(path):
            sys.exit(f"File not found: {path}")

    with open(args.classes) as f:
        classes = json.load(f)
    model = keras.models.load_model(args.model)

    ensure_hand_model()
    options = mp.tasks.vision.HandLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=HAND_MODEL_FILE),
        running_mode=mp.tasks.vision.RunningMode.VIDEO,
        num_hands=1,
    )
    detector = mp.tasks.vision.HandLandmarker.create_from_options(options)

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        sys.exit("Could not open the webcam. Try --camera 1.")

    recent = deque(maxlen=args.smooth)   # last few frame-level predictions
    text = ""                            # the word/sentence typed so far
    hold_label, hold_start, committed = None, 0.0, False
    start_time, last_ts = time.time(), -1

    print("Running. Click the video window, then press q to quit.")
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if args.flip_input:
            frame = cv2.flip(frame, 1)

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        ts = int((time.time() - start_time) * 1000)
        ts = max(ts, last_ts + 1)        # MediaPipe needs strictly increasing timestamps
        last_ts = ts
        result = detector.detect_for_video(mp_image, ts)

        label, conf = None, 0.0
        if result.hand_landmarks:
            hand = result.hand_landmarks[0]
            draw_hand(frame, hand)
            feats = landmarks_to_features(hand)
            if feats is not None:
                probs = model(feats.reshape(1, -1), training=False).numpy()[0]
                best = int(np.argmax(probs))
                if probs[best] >= args.min_conf:
                    label, conf = classes[best], float(probs[best])
        recent.append(label)

        # Majority vote over the last few frames to stop the label flickering.
        top, count = Counter(recent).most_common(1)[0]
        stable = top if (top is not None and count >= 0.7 * args.smooth) else None

        # A sign is typed once it has been held steady for --hold seconds.
        now = time.time()
        progress = 0.0
        if stable is None:
            hold_label, committed = None, False
        elif stable != hold_label:
            hold_label, hold_start, committed = stable, now, False
        elif not committed:
            progress = min((now - hold_start) / args.hold, 1.0)
            if progress >= 1.0:
                if stable == "space":
                    text += " "
                elif stable == "del":
                    text = text[:-1]
                else:
                    text += stable
                committed = True     # lower your hand before typing the same sign again

        # ---- on-screen display ----
        h, w = frame.shape[:2]
        shown = f"{stable} ({conf * 100:.0f}%)" if stable else "show a sign"
        cv2.putText(frame, shown, (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
        cv2.rectangle(frame, (10, 48), (10 + int(160 * progress), 58), (0, 255, 0), -1)
        cv2.rectangle(frame, (10, 48), (170, 58), (255, 255, 255), 1)
        cv2.rectangle(frame, (0, h - 50), (w, h), (0, 0, 0), -1)
        cv2.putText(frame, "Text: " + text, (10, h - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

        cv2.imshow("Sign Language Recognition", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key == ord("c"):
            text = ""
        if key == ord("b"):
            text = text[:-1]

    cap.release()
    cv2.destroyAllWindows()
    detector.close()
    print("Final text:", text)


if __name__ == "__main__":
    main()
