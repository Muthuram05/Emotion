import cv2
from deepface import DeepFace
from playsound import playsound

neutral = 0
surprise = 0
sad = 0
happy =0 
fear = 0
disgust = 0
angry = 0 

cap = cv2.VideoCapture(1)
if not cap.isOpened():
        cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise IOError("No camera detected")

def fun():
    if (angry == 1):
        playsound("music/angry.mp3")
    elif (disgust == 1):
        playsound("music/disgust.mp3")
    elif (fear == 1):
        playsound("music/fear.mp3")
    elif (happy == 1):
        playsound("music/happy.mp3")
    elif (neutral == 1):
        playsound("music/neutral.mp3")
    elif (sad == 1):
        playsound("music/sad.mp3")
    elif (surprise == 1):
        playsound("music/surprise.mp3")
    else:
        pass

while True:
    ret, frame = cap.read()
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    result_analyzer = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
    font = cv2.FONT_HERSHEY_PLAIN
    cv2.putText(frame, result_analyzer['dominant_emotion'], (50, 50), font, 3, (0, 0, 255), 2, cv2.LINE_4)
    cv2.imshow('Original video', frame)
    if (result_analyzer["dominant_emotion"] == 'angry'):
        angry = 1
        disgust,fear,happy,sad,surprise,neutral = 0,0,0,0,0,0
    elif (result_analyzer["dominant_emotion"] == 'disgust'):
        disgust=1
        angry,fear,happy,sad,surprise,neutral = 0,0,0,0,0,0
    elif (result_analyzer["dominant_emotion"] == 'fear'):
        fear = 1
        disgust,angry,happy,sad,surprise,neutral = 0,0,0,0,0,0 
    elif (result_analyzer["dominant_emotion"] == 'happy'):
        happy = 1
        disgust,fear,angry,sad,surprise,neutral = 0,0,0,0,0,0
    elif (result_analyzer["dominant_emotion"] == 'sad'):
        sad = 1
        disgust,fear,happy,angry,surprise,neutral = 0,0,0,0,0,0 
    elif (result_analyzer["dominant_emotion"] == 'surprise'):
        surprise = 1
        disgust,fear,happy,sad,angry,neutral = 0,0,0,0,0,0 
    elif (result_analyzer["dominant_emotion"] == 'neutral'):
        neutral = 1
        disgust,fear,happy,sad,surprise,angry = 0,0,0,0,0,0 
    else:
        pass
    fun()
    if cv2.waitKey(2) & 0xFF == ord('q'):
        break


