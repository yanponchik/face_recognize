import cv2

import time
import numpy as np





face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

cv2.startWindowThread()
cap = cv2.VideoCapture(0)

out = cv2.VideoWriter(
    'output.avi',
    cv2.VideoWriter_fourcc(*'MJPG'),
    15.,
    (640, 480))

def PRINTFRAME(frame):
    cv2.imshow('img', frame)
    out.write(frame.astype('uint8'))

def Photo():
    img, frame = cap.read()

    frame = cv2.resize(frame, (640, 480))

    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

    return gray, frame

def FACES(gray, frame):

    faces = face_cascade.detectMultiScale(gray, 1.1, 4)

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

def Human(frame):

    boxes, weights = hog.detectMultiScale(frame, winStride=(8, 8))

    boxes = np.array([[x, y, x + w, y + h] for (x, y, w, h) in boxes])

    for (xA, yA, xB, yB) in boxes:

        cv2.rectangle(frame, (xA, yA), (xB, yB), (0, 255, 0), 2)

while True:
    start = time.time()
    gray_get, frame_get = Photo()

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break

    FACES(gray_get, frame_get)
    Human(frame_get)
    PRINTFRAME(frame_get)
    end = time.time()
    print("The time of execution of above program is :", end-start)

cap.release()
out.release()
cv2.destroyAllWindows()
cv2.waitKey(1)


# difference of start and end variables
# gives the time of execution of the
# program in between
