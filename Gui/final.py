import cv2
from deepface import DeepFace
from playsound import playsound

cam_port = 0
cam = cv2.VideoCapture(cam_port)
  
# reading the input using the camera

while True:
        result, image = cam.read()
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        result_analyzer = DeepFace.analyze(image, actions=['emotion'], enforce_detection=False)
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)
        for (x, y, w, h) in faces:
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        font = cv2.FONT_HERSHEY_PLAIN
        cv2.putText(image, result_analyzer['dominant_emotion'], (50, 50), font, 3, (0, 0, 255), 2, cv2.LINE_4)
        cv2.imshow('Original video', image)
        key = cv2.waitKey(1)
        if key == ord('s'): 
            cv2.imwrite("Final.png", image)
            result_analyzer = DeepFace.analyze('Final.png', actions=['emotion'], enforce_detection=False)
            res = result_analyzer['dominant_emotion']
            print(res)
            playsound("music/"+ res+ ".mp3")

            


