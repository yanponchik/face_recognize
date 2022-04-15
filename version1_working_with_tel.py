import numpy as np
import cv2
import os
import telebot
from threading import Thread
bot = telebot.TeleBot('5267233996:AAHbj2pDpg4gBh4BBJoAb7mxylv3lOAU7DI')

#template = cv2.imread('cam.png',0)
cv2.startWindowThread()
cap = cv2.VideoCapture(0)

#template = template[80:380, 180:470]
#w, h = template.shape[::-1]
a = 0

@bot.message_handler(commands=['start'])
def welcome(message):
    bot.send_message(message.chat.id, 'yan lox')

@bot.message_handler(content_types=['text'])
def message_test(message):
    words = ['фото']
    for word in words:
        if word in message.text:
            while True:
                bot.send_photo(message.chat.id, open('cam3.png', 'rb'))
        else:
            bot.send_message(message.chat.id, 'hfgjfhjg')
      

def pool():
    bot.polling()
th_2 = Thread(target=pool, args=())
th_2.start()


th_3 = Thread(target=message_test, args=())
th_3.start()


th_4 = Thread(target=welcome, args=())
th_4.start()

b = 0
while True:
    if b == 1:
        cv2.imwrite('cam3.png', frame)
        b = 0
    b = 0 
    # Capture frame-by-frame
    ret, frame = cap.read()
    dicton = 'faces'
    files = os.listdir(dicton)
    for i in files:
        print(i)
        template = cv2.imread('./faces/' + i, 0)
        template = template[80:380, 180:470]
        w, h = template.shape[::-1]
    # Our operations on the frame come here
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        res = cv2.matchTemplate(gray,template,cv2.TM_CCOEFF_NORMED)
        threshold = 0.65
        loc = np.where( res >= threshold)
        a = 0
        #print(res)
        for pt in zip(*loc[::-1]):
            cv2.rectangle(gray, pt, (pt[0] + w, pt[1] + h), (0,0,255), 2)
            a = 1
        if a == 1:
            print('lica')
        else:
            print('net lic')
            b = 1
        cv2.imshow('frame',gray)
        cv2.imshow('temp',template)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    if b == 1:
        cv2.imwrite('cam3.png', frame)
        b = 0
        
        
    

        
      
            
        # Display the resulting frame
    

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
