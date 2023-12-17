import cv2
import numpy as np
from keras.models import load_model
import pygame

# Load model
model = load_model('preprocessingdata1.h5')

# Load cascade classifiers for face and eyes
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

# Initialize pygame mixer
pygame.mixer.init()

# Load alarm sound
alarm_filename = 'alarm.wav'
alarm_sound = pygame.mixer.Sound(alarm_filename)

# Function to play alarm
def play_alarm():
    alarm_sound.play()

# Function to detect drowsiness
def detect_drowsiness(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        roi_gray = gray[y:y + h, x:x + w]
        eyes = eye_cascade.detectMultiScale(roi_gray)

        for (ex, ey, ew, eh) in eyes:
            roi_eye = roi_gray[ey:ey + eh, ex:ex + ew]
            
            # Resize the eye region to match the model's expected input size
            roi_eye = cv2.resize(roi_eye, (256, 256))
            
            # Convert to RGB (if the model expects RGB input)
            roi_eye = cv2.cvtColor(roi_eye, cv2.COLOR_GRAY2RGB)
            
            roi_eye = roi_eye.astype("float") / 255.0
            roi_eye = np.expand_dims(roi_eye, axis=0)

            prediction = model.predict(roi_eye)[0][0]
            if prediction < 0.1:
                cv2.putText(frame, "Drowsy", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                play_alarm()
            else:
                cv2.putText(frame, "Awake", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    return frame

# Capture video from the webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()

    if not ret:
        break

    frame = detect_drowsiness(frame)

    cv2.imshow("Drowsiness Detection", frame)

    if cv2.waitKey(1) & 0xFF == 27:  # Press 'Esc' to exit
        break

cap.release()
cv2.destroyAllWindows()