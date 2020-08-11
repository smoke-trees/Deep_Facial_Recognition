# -*- coding: utf-8 -*-
"""
Created on Fri Dec 13 00:55:13 2019

@author: Unnikrishnan Menon
"""

import cv2
import imutils
from imutils.video import VideoStream
import os 


def image_capture(name):
    n=0
    print('Press C to Capture...')
    detector=cv2.CascadeClassifier('haarcascade_frontalface_alt2.xml')
    vid=VideoStream(src=0).start()
    while True:
        frame=vid.read()
        temp=frame.copy()
        frame=imutils.resize(frame, width=400)
        rects=detector.detectMultiScale(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        for (x, y, w, h) in rects:
            cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)    
        cv2.imshow("Facial Recognition", frame)
        key=cv2.waitKey(1) & 0xFF
        if key==ord("c"):
            cv2.imwrite("./images/"+name+".jpg",frame[y:y+h,x:x+w])
            n+=1
        elif key==ord("q") or n==1:
            break        
    print("{} Image has been stored in the dataset!".format(n))
    cv2.destroyAllWindows()
    vid.stop()

name = input('Enter the name of the person to Enroll : ')
image_capture(name)
