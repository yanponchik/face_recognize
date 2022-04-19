import numpy as np
import cv2
import os
import telebot
import shutil
from threading import Thread
bot = telebot.TeleBot('5267233996:AAHbj2pDpg4gBh4BBJoAb7mxylv3lOAU7DI')
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

cv2.startWindowThread()
cap = cv2.VideoCapture(0)

schet_lic_cascad = 0
schet_lic_algoritm = 0
a = 0


@bot.message_handler(commands=['start'])
def welcome(message):
    bot.send_message(message.chat.id, 'strating')

@bot.message_handler(content_types=['text'])
def message_test(message):
    words = ['фото', 'список']
    if 'фото' in message.text:
        while message.text != 'стоп':
            bot.send_photo(message.chat.id, open('cam3.png', 'rb'))
    elif 'список' in message.text:
        dicton2 = 'list_foto'
        files2 = os.listdir(dicton2)
        for i in files2:
            bot.send_photo(message.chat.id, open('./list_foto/' + i, 'rb'))
        
    else:
        bot.send_message(message.chat.id, 'неверно')


def pool():
    bot.polling()
th_2 = Thread(target=pool, args=())
th_2.start()


th_3 = Thread(target=message_test, args=())
th_3.start()


th_4 = Thread(target=welcome, args=())
th_4.start()

b = 0
def PRINTFRAME(frame):
    cv2.imshow('img', frame)

def Photo():
    img, frame = cap.read()

    frame = cv2.resize(frame, (640, 480))

    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

    return gray, frame
yan = 0

def FACES(gray, frame):
    global yan
    yan = 0
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        print('lica_img')
        yan = 1

r = 0
while True:
    r += 1
    print(r)
    b = 0
    gray_get, frame_get = Photo()
    ret, frame = cap.read()
    FACES(gray_get, frame_get)
    PRINTFRAME(frame_get)
    dicton = 'faces'
    files = os.listdir(dicton)
    for i in files:
        template = cv2.imread('./faces/' + i, 0)
        template = template[80:380, 180:470]
        w, h = template.shape[::-1]

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        res = cv2.matchTemplate(gray,template,cv2.TM_CCOEFF_NORMED)
        threshold = 0.65
        loc = np.where( res >= threshold)
        a = 0
        for pt in zip(*loc[::-1]):
            cv2.rectangle(gray, pt, (pt[0] + w, pt[1] + h), (0,0,255), 2)
            a = 1
        if a == 1:
            print('lica')
            b = 1
        else:
            print('lica ne sovpali')
        cv2.imshow('frame',gray)
        cv2.imshow('temp',template)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    if b == 0 and yan == 1:
        r_1 = str(r)
        
        cv2.imwrite('cam3.png', frame)
        if r <= 100:
            f = open('t' + r_1 + '.png', 'w')
            cv2.imwrite('t' + r_1 + '.png', frame)
            source = 't' + r_1 + '.png'
            destination = 'list_foto'
            os.replace(source, 'list_foto/t' + r_1 + '.png')
        b = 0
        

cap.release()
cv2.destroyAllWindows()
