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
import imutils #user made library that make it easier to do basic task
import time

cap =cv2.VideoCapture(0)

while (True):
    #Capture frame by frame
    ret, frame = cap.read(0)

    #Our operations on the frame come here
    frame = imutils.resize(frame, width=600) #downsize so fps can stay higher
    blurred = cv2.GaussianBlur(frame, (11, 11), 0) #reduce noise
    HSV =cv2.cvtColor(frame.copy(), cv2.COLOR_BGR2HSV) #HSV conversion


    # define range of yellow color in HSV yellow is 30,255,255
    lower_yellow = (20, 80, 5)
    upper_yellow = (65, 255, 255)

     # Threshold the HSV image to get only blue colors
    mask = cv2.inRange(HSV , lower_yellow, upper_yellow)
    mask = cv2.erode(mask, None, iterations=2) #remove blobs
    mask = cv2.dilate(mask, None, iterations=2)

    # Bitwise-AND mask and original image
    #res = cv2.bitwise_and(frame ,frame, mask= mask)
    #circles= cv2.HoughCircles(mask, cv2.HOUGH_GRADIENT, 1, 260, param1=30, param2=65, minRadius=0, maxRadius=0)
    #print(circles)

    #more advanced function than hough circles since part of circle goes sometimes
    cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,
		cv2.CHAIN_APPROX_SIMPLE) 

    #if circles is not None:
	# convert the (x, y) coordinates and radius of the circles to integers
	#circles = np.round(circles[0, :]).astype("int")
 
	# loop over the (x, y) coordinates and radius of the circles
	#for (x, y, r) in circles:
		# draw the circle in the output image, then draw a rectangle
		# corresponding to the center of the circle
		#cv2.circle(mask, (x, y), r, (0, 255, 0), 4)
		#cv2.rectangle(output, (x - 5, y - 5), (x + 5, y + 5), (0, 128, 255), -1)
 
   
    cnts = cnts[1]
    center = None
    print(cnts)
	# only proceed if at least one contour was found
    if len(cnts) > 0:
        #find largest contour
		c = max(cnts, key=cv2.contourArea)

        #find minimum enclosing circle around contour
		((x, y), radius) = cv2.minEnclosingCircle(c)

        #Computation of centroid
		M = cv2.moments(c)
		center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
 
		# only proceed if the radius meets a minimum size
		if radius > 1:
			# draw the circle on the frame,
			cv2.circle(frame, (int(x), int(y)), int(radius),(0, 255, 255), 2)
	

    #Display the resulting frame
    cv2.imshow('frame',frame)
    #cv2.imshow('H: Intensity', HSV[:,:,0])
    #cv2.imshow('S: Saturation', HSV[:,:,1])
    #v2.imshow('V:Value', HSV[:,:,2])
    cv2.imshow('mask',mask)
    #cv2.imshow('circle',np.hstack([frame, mask]))
    #cv2.imshow('res',res)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# method to look for hough circles circles = cv2.HoughCircles(cv_image, cv2.HOUGH_GRADIENT, 1, 90, param1=70, param2=60,minRadius=50, maxRadius=0)

cap.release()
cv2.destroyAllWindows()