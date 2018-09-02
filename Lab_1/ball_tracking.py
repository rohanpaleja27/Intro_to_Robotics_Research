#!/usr/bin/env python

# HELP LINKS/RESOURCE
#http://anikettatipamula.blogspot.com/2012/12/ball-tracking-detection-using-opencv.html
#https://www.pyimagesearch.com/2015/09/14/ball-tracking-with-opencv/


# Code for checking HSV value of an RGB color
#>>> green = np.uint8([[[0,255,0 ]]])
#>>> hsv_green = cv2.cvtColor(green,cv2.COLOR_BGR2HSV)
#>>> print hsv_green

import cv2
import numpy as np 

cap =cv2.VideoCapture(0)

while (True):
    #Capture frame by frame
    ret, frame = cap.read(0)

    #Our operations on the frame come here
    HSV =cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)


    # define range of yellow color in HSV yellow is 30,255,255
    lower_yellow = np.array([0,0,0])
    upper_yellow = np.array([50,255,255])

     # Threshold the HSV image to get only blue colors
    mask = cv2.inRange(HSV , lower_yellow, upper_yellow)

    # Bitwise-AND mask and original image
    res = cv2.bitwise_and(frame ,frame, mask= mask)

    #Display the resulting frame
    cv2.imshow('frame',HSV)
    cv2.imshow('mask',mask)
    cv2.imshow('res',res)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# method to look for hough circles circles = cv2.HoughCircles(cv_image, cv2.HOUGH_GRADIENT, 1, 90, param1=70, param2=60,minRadius=50, maxRadius=0)

cap.release()
cv2.destroyAllWindows()