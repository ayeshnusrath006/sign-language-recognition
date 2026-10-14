# Real-Time Sign Language Recognition (ASL Alphabet)

Elevate Labs AI & ML Internship (MSME, Govt. of India) - Project Phase, Project 7.

A system that recognises American Sign Language (ASL) alphabet signs from hand images and from a live webcam, and combines recognised letters into words.

## How it works

1. **Hand tracking:** MediaPipe's `HandLandmarker` finds 21 landmark points on the hand (each with x, y, z).
2. **Feature preparation:** the 21 points are made relative to the wrist and divided by the largest distance, giving 63 numbers that do not depend on where the hand is in the picture or how big it looks.
3. **Classification:** a small Keras neural network (Dense 128 -> Dropout -> Dense 64 -> Dropout -> Softmax) predicts the sign from those 63 numbers.
4. **Real-time use:** frames from the webcam go through the same steps. A sign is accepted only when the prediction is confident and stays the same for about 1.2 seconds, and accepted letters are appended to an on-screen text line. The `space` sign adds a space and the `del` sign deletes the last character.

I used landmark coordinates instead of raw images for the classifier. This makes training fast and makes the model less sensitive to background and lighting.

## Dataset

[ASL Alphabet](https://www.kaggle.com/datasets/grassknoted/asl-alphabet) (Kaggle, `grassknoted`): 3000 images for each of 29 classes (A-Z, `del`, `nothing`, `space`).

- The `nothing` class (no hand) is not used, because in live use "no hand detected" already means nothing is being signed.
- 500 random images per class were processed (14,000 images). A hand was detected in 10,626 of them, and those became the training data (28 classes).
- Split: 80% training, 20% testing (stratified).

## Results

| Metric | Value (after 30 epochs) |
|---|---|
| Training accuracy | 96.06% |
| Test accuracy | 97.55% |
| Test loss | 0.098 |

Training curves are in `training_curves.png`. Test accuracy is slightly higher than training accuracy because dropout is only active during training.

## Files

| File | Purpose |
|---|---|
| `sign_language_recognition.ipynb` | Google Colab notebook: data download, landmark extraction, training, evaluation, saving |
| `realtime_sign_recognition.py` | Live webcam recognition script with word building |
| `sign_model.keras` | Trained model |
| `classes.json` | Class names in the order the model uses |
| `landmarks.npz` | Extracted landmark features and labels used for training |
| `training_curves.png` | Accuracy and loss curves |
| `requirements.txt` | Python packages |

## How to run

### Train (Google Colab)
1. Open `sign_language_recognition.ipynb` in [Google Colab](https://colab.research.google.com).
2. Create a Kaggle API token (kaggle.com -> Settings -> API) and paste it when the notebook asks.
3. Run the cells in order.

### Real-time webcam (on a computer with a webcam)
```bash
pip install -r requirements.txt
python realtime_sign_recognition.py
```
Keep `sign_model.keras` and `classes.json` in the same folder. The MediaPipe hand model file is downloaded automatically on first run.

Options:
- `--flip-input` mirrors the camera image before detection. Try it if predictions are consistently wrong (for example, if you sign with the other hand).
- `--hold 1.2` seconds a sign must be held to be typed.
- `--min-conf 0.80` minimum confidence to accept a prediction.
- `--camera 1` choose another camera.

Keys (click the video window first): `q` quit, `c` clear text, `b` delete last character. To type the same letter twice, lower your hand briefly between the two.

## Tools used

Python, MediaPipe (HandLandmarker), TensorFlow/Keras, OpenCV, NumPy, scikit-learn, Matplotlib, Google Colab, Kaggle.

## Limitations

- The dataset's photos come from a small number of signers and a similar setting, so accuracy with a different person, camera or lighting can be lower than the test-set figure.
- The letters J and Z involve motion in real ASL; this project classifies single frames only.
- Only one hand is tracked, and the model recognises static alphabet signs, not full sign language.
