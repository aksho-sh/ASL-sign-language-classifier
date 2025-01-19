import cv2
import torch
import mediapipe as mp
import numpy as np
import joblib
import traceback
import logging

logging.basicConfig(level=logging.DEBUG)

class GestureClassifier(torch.nn.Module):
    def __init__(self, input_size, num_classes):
        super(GestureClassifier, self).__init__()
        self.fc = torch.nn.Sequential(
            torch.nn.Linear(input_size, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, num_classes)
        )

    def forward(self, x):
        return self.fc(x)
    
print ("reached 1")

model = GestureClassifier(input_size=63, num_classes=29)
try:
    print("reached here 1.1")
    model.load_state_dict(torch.load("./gesture_classifier.pth", map_location=torch.device('cpu')))
    print("reached here 1.2")
    model.eval()
    print("reched here 1.3")
    logging.info("Model loaded successfully")
except Exception as e:
    logging.error("Error loading the model:")
    logging.error(traceback.format_exc())
    exit()
print ("reached 2")

try:
    label_mapping = joblib.load("./label_mapping.pkl")
    logging.info("Label mapping loaded successfully")
except Exception as e:
    logging.error("Error loading label mapping:")
    logging.error(traceback.format_exc())
    exit()
print ("reached 3")

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5)

def preprocess_landmarks(landmarks):
    """
    Preprocess Mediapipe landmarks for model inference.
    """
    flattened_landmarks = [coord for lm in landmarks for coord in (lm.x, lm.y, lm.z)]
    return torch.tensor(flattened_landmarks, dtype=torch.float32).unsqueeze(0)  # Add batch dimension
print ("reached 4")

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    logging.error("Error: Could not open webcam.")
    exit()

logging.info("Starting real-time inference... Press 'q' to quit.")
print ("reached 5")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        logging.warning("Failed to grab frame.")
        break

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            input_tensor = preprocess_landmarks(hand_landmarks.landmark)
            with torch.no_grad():
                output = model(input_tensor)
                _, predicted = torch.max(output, 1)
                predicted_label = [key for key, value in label_mapping.items() if value == predicted.item()][0]

            mp.solutions.drawing_utils.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            cv2.putText(frame, f"Prediction: {predicted_label}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    cv2.imshow('Real-Time Gesture Recognition', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
hands.close()
cv2.destroyAllWindows()
