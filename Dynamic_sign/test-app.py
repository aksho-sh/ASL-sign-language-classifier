import cv2
import mediapipe as mp
import numpy as np
import torch
import torch.nn as nn
import os
import time
from typing import List, Optional, Tuple
from collections import deque

class ImprovedLandmarkLSTM(nn.Module):
    def __init__(
        self, 
        input_size=63,
        hidden_size=256,
        num_layers=3,
        num_classes=3,
        dropout_rate=0.4
    ):
        super(ImprovedLandmarkLSTM, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout_rate if num_layers > 1 else 0,
            bidirectional=True
        )
        
        self.fc = nn.Sequential(
            nn.Linear(hidden_size * 2, 256),
            nn.BatchNorm1d(256),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        batch_size = x.size(0)
        h0 = torch.zeros(self.num_layers * 2, batch_size, self.hidden_size, device=x.device)
        c0 = torch.zeros(self.num_layers * 2, batch_size, self.hidden_size, device=x.device)
        
        out, _ = self.lstm(x, (h0, c0))
        out = torch.cat((out[:, -1, :self.hidden_size], out[:, 0, self.hidden_size:]), dim=1)
        out = self.fc(out)
        return out

class ProductionGestureDetector:
    def __init__(
        self,
        model_path: str,
        device: str = "cpu",
        sequence_length: int = 128,
        image_size: Tuple[int, int] = (128, 128),
        target_height: int = 480
    ):
        """Initialize production gesture detector"""
        self.device = torch.device(device)
        self.sequence_length = sequence_length
        self.image_size = image_size
        self.target_height = target_height
        self.classes = ['hello', 'negative', 'thank_you']
        self.debug_mode = False
        
        # Initialize MediaPipe
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.5
        )
        
        # Load model
        print(f"\nLoading model from {model_path}...")
        self.model = self._load_model(model_path)
        print("Model loaded successfully")
        
        # Message queue for displaying temporary messages
        self.message_queue = deque(maxlen=5)
        self.message_duration = 3  # seconds
        self.message_timestamps = deque(maxlen=5)
        
        # Status variables
        self.last_prediction = None
        self.last_confidence = None
        self.recording_start_time = None
        
        # Ensure output directories exist
        os.makedirs("recorded_videos", exist_ok=True)
        os.makedirs("detected_signs", exist_ok=True)

    def _load_model(self, model_path: str) -> nn.Module:
        """Load the trained model"""
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
            
        checkpoint = torch.load(model_path, map_location=self.device)
        model = ImprovedLandmarkLSTM(
            input_size=63,
            hidden_size=256,
            num_layers=3,
            num_classes=len(self.classes),
            dropout_rate=0.4
        )
        
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(self.device)
        model.eval()
        return model

    def _resize_to_480p(self, frame):
        """Resize frame to 480p height maintaining aspect ratio"""
        height = self.target_height
        aspect_ratio = frame.shape[1] / frame.shape[0]
        width = int(height * aspect_ratio)
        width = width + (width % 2)
        return cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)

    def add_message(self, message: str):
        """Add a temporary message to the queue"""
        self.message_queue.append(message)
        self.message_timestamps.append(time.time())

    def _draw_ui(self, frame: np.ndarray, is_recording: bool, frame_count: int = 0) -> np.ndarray:
        """Draw all UI elements on the frame"""
        height, width = frame.shape[:2]
        
        # Create semi-transparent overlay for UI background
        overlay = frame.copy()
        
        # Draw main status panel background
        cv2.rectangle(overlay, (10, 10), (width-10, 150), (0, 0, 0), -1)
        
        # Draw controls panel background
        cv2.rectangle(overlay, (10, height-140), (width-10, height-10), (0, 0, 0), -1)
        
        # Blend overlay with main frame
        alpha = 0.7
        frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)
        
        # Draw recording status
        status_color = (0, 0, 255) if is_recording else (255, 255, 255)
        cv2.putText(
            frame,
            f"{'🔴 Recording' if is_recording else '⚪ Ready to Record'}",
            (30, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            status_color,
            2
        )
        
        # Draw recording time if recording
        if is_recording and self.recording_start_time is not None:
            elapsed_time = time.time() - self.recording_start_time
            cv2.putText(
                frame,
                f"Time: {elapsed_time:.1f}s | Frames: {frame_count}",
                (30, 70),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2
            )
        
        # Draw last prediction if available
        if self.last_prediction is not None:
            confidence_color = (0, 255, 0) if self.last_confidence > 0.8 else (0, 165, 255)
            cv2.putText(
                frame,
                f"Last Prediction: {self.last_prediction}",
                (30, 100),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                confidence_color,
                2
            )
            cv2.putText(
                frame,
                f"Confidence: {self.last_confidence:.2f}",
                (30, 130),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                confidence_color,
                2
            )
        
        # Draw controls
        controls = [
            "R - Start/Stop Recording",
            "P - Predict Last Recording",
            "D - Toggle Debug Mode",
            "Q - Quit"
        ]
        
        for i, control in enumerate(controls):
            cv2.putText(
                frame,
                control,
                (30, height-110 + i*30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2
            )
        
        # Draw debug status if enabled
        if self.debug_mode:
            cv2.putText(
                frame,
                "Debug Mode ON",
                (width-200, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2
            )
        
        # Draw temporary messages
        current_time = time.time()
        while self.message_timestamps and \
              current_time - self.message_timestamps[0] > self.message_duration:
            self.message_queue.popleft()
            self.message_timestamps.popleft()
        
        for i, message in enumerate(self.message_queue):
            cv2.putText(
                frame,
                message,
                (30, height//2 + i*30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 255),
                2
            )
        
        return frame

    def _process_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
        """Process frame consistently with training pipeline"""
        # 1. Flip frame horizontally first (matching video-collector)
        frame = cv2.flip(frame, 1)
        
        # 2. Make a copy for visualization
        viz_frame = frame.copy()
        
        # 3. Process at original resolution first (matching training)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(frame_rgb)
        
        # 4. Extract landmarks at original resolution
        landmarks = None
        if results.multi_hand_landmarks:
            landmarks = []
            for landmark in results.multi_hand_landmarks[0].landmark:
                landmarks.extend([landmark.x, landmark.y, landmark.z])
            landmarks = np.array(landmarks, dtype=np.float32)
            
            # Draw landmarks on visualization frame
            self.mp_drawing.draw_landmarks(
                viz_frame,
                results.multi_hand_landmarks[0],
                self.mp_hands.HAND_CONNECTIONS
            )
        else:
            landmarks = np.zeros(21 * 3, dtype=np.float32)
        
        # 5. Resize frame to 480p maintaining aspect ratio
        processed_frame = self._resize_to_480p(frame)
        
        return viz_frame, processed_frame, landmarks

    def start_recording(self):
        """Start webcam recording with controls"""
        print("\nStarting webcam capture...")
        
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            raise ValueError("Could not open webcam")
        
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        is_recording = False
        recorded_frames = []
        recorded_landmarks = []
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Process frame using consistent pipeline
                viz_frame, processed_frame, landmarks = self._process_frame(frame)
                
                # Draw UI elements
                display_frame = self._draw_ui(
                    viz_frame, 
                    is_recording, 
                    len(recorded_frames)
                )
                
                # Handle recording
                if is_recording:
                    recorded_frames.append(processed_frame)
                    recorded_landmarks.append(landmarks)
                
                # Display frame
                cv2.imshow('ASL Gesture Recognition', display_frame)
                
                # Handle key presses
                key = cv2.waitKey(1) & 0xFF
                
                if key == ord('r'):
                    is_recording = not is_recording
                    if is_recording:
                        self.recording_start_time = time.time()
                        recorded_frames = []
                        recorded_landmarks = []
                        self.add_message("Started recording...")
                    else:
                        self.recording_start_time = None
                        if recorded_frames:
                            timestamp = time.strftime("%Y%m%d_%H%M%S")
                            video_path = os.path.join("recorded_videos", f"recording_{timestamp}.mp4")
                            
                            height, width = recorded_frames[0].shape[:2]
                            out = cv2.VideoWriter(
                                video_path,
                                cv2.VideoWriter_fourcc(*'mp4v'),
                                fps,
                                (width, height)
                            )
                            
                            for f in recorded_frames:
                                out.write(f)
                            out.release()
                            self.add_message(f"Saved recording: {video_path}")
                            
                elif key == ord('p'):
                    if not is_recording and recorded_landmarks:
                        self.add_message("Processing recorded sequence...")
                        prediction, confidence = self._process_landmark_sequence(recorded_landmarks)
                        self.last_prediction = prediction
                        self.last_confidence = confidence
                        self.add_message(f"Prediction: {prediction} ({confidence:.2f})")
                
                elif key == ord('d'):
                    self.debug_mode = not self.debug_mode
                    self.add_message(f"Debug mode: {'ON' if self.debug_mode else 'OFF'}")
                    
                elif key == ord('q'):
                    break
                    
        finally:
            cap.release()
            cv2.destroyAllWindows()

    def _process_landmark_sequence(self, landmark_sequence: List[np.ndarray]) -> Tuple[str, float]:
        """Process landmark sequence and make prediction"""
        landmark_sequence = np.array(landmark_sequence)
        
        # Handle sequence length
        if len(landmark_sequence) > self.sequence_length:
            # Take middle sequence
            start = (len(landmark_sequence) - self.sequence_length) // 2
            landmark_sequence = landmark_sequence[start:start + self.sequence_length]
        else:
            # Pad with zeros
            padding = np.zeros((self.sequence_length - len(landmark_sequence), 21 * 3), dtype=np.float32)
            landmark_sequence = np.concatenate([landmark_sequence, padding])
        
        # Make prediction
        sequence_tensor = torch.FloatTensor(landmark_sequence).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(sequence_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            confidence, predicted = torch.max(probabilities, 1)
        
        prediction = self.classes[predicted.item()]
        confidence = confidence.item()
        
        return prediction, confidence

def main():
    try:
        print("\nInitializing ASL Gesture Detector...")
        detector = ProductionGestureDetector(
            model_path='Model/asl_landmark_model.pth'
        )
        
        print("\nStarting recording interface...")
        detector.start_recording()
        
    except FileNotFoundError as e:
        print(f"\nError: File not found - {str(e)}")
    except Exception as e:
        print(f"\nError: {str(e)}")
        import traceback
        traceback.print_exc()
    finally:
        print("\nCleaning up...")
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()