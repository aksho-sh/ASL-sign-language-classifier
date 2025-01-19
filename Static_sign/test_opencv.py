import cv2

print(cv2.__version__)

camera = cv2.VideoCapture(0)

while True:
    
    status, frame  = camera.read()
    if not status:
        break
    
    cv2.imshow('Webcam Feed', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
camera.release()
cv2.destroyAllWindows()