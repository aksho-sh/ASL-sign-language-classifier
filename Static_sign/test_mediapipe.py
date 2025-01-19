import cv2
import mediapipe as mp

media_hands = mp.solutions.hands

hands = media_hands.Hands(min_detection_confidence = 0.7, min_tracking_confidence = 0.7, static_image_mode = False, max_num_hands=2)

mp_drawing = mp.solutions.drawing_utils

camera = cv2.VideoCapture(0)

while camera.isOpened():
    
    status, frame = camera.read()
    if not status:
        break
    
    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb_frame)
    
    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, media_hands.HAND_CONNECTIONS)
    
    cv2.imshow('Hand Traking', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
camera.release()
cv2.destroyAllWindows()
