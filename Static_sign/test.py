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
    
model = GestureClassifier(input_size=63, num_classes=29)

import traceback

try:
    model.load_state_dict(torch.load("./gesture_classifier.pth", map_location=torch.device('cpu')))
    model.eval()
    print("Model loaded successfully.")
except Exception as e:
    print("Error loading model:")
    print(e.args)
    exit(1)

print(model)