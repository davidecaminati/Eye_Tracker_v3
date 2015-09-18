

# import the necessary packages
from pyimagesearch.eyetracker_no_face import Eyetracker_no_face
from pyimagesearch import imutils

#from skimage.filter import threshold_adaptive
import numpy as np

import argparse
import time
import cv2
from collections import Counter


def Fase1(camera,et,Debug,eye_to_track):
	
	number = 0
	rectArray = []
	number_common_rect = 0
	b = Counter(rectArray)
	while number_common_rect < 1: #at least 3 entry of the same rect

		start = time.time()
		
		#find the must common rect
		if len(rectArray) > 5:
			b = Counter(rectArray)
			if Debug:
				print "b.most_common(1)" + str(b.most_common(1))
				print "number = " + str(b.most_common(1)[0][1])
			number_common_rect = b.most_common(1)[0][1]
			
		(grabbed, frame) = camera.read()
		# grab the raw NumPy array representing the image
		
		# check to see if we have reached the end of the
		# video
		if not grabbed:
			break
		
		# resize the frame and convert it to grayscale
		#frame = imutils.resize(frame, width = 300)
		gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
		gray = cv2.blur(gray, (3,3))
		
		# detect eyes in the image
		rects = et.track(gray)
		
		# loop over the eyes bounding boxes and draw them
		for rect in rects:
			(h, w) = frame.shape[:2]
			if Debug:
				print rect[0],h,w
			cv2.rectangle(frame, (rect[0], rect[1]), (rect[2], rect[3]), (0, 255, 0), 2)
			r0 = rect[0]
			r1 = rect[1]
			r2 = rect[2]
			r3 = rect[3]
			if eye_to_track == "0": # Left eye
				if rect[0] <= w/2:
					number += 1
					if Debug:
						print "left"
					rectArray.append(rect)
			else:                   # Right eye
				if rect[0] >= w/2:
					number += 1
					if Debug:
						print "right"
					rectArray.append(rect)
			
		
		# show the tracked eyes 
		if Debug:
			cv2.imshow("Tracking", frame)
		# clear the frame in preparation for the next frame
		#rawCapture.truncate(0)
		
		# calcolate performance
		end = time.time()
		difference = end - start
		
		if Debug:
			print difference
		##SendToArduino(difference)
		
		# if the 'q' key is pressed, stop the loop
		if cv2.waitKey(1) & 0xFF == ord("q"):
			break
	return rectArray,number,rect,b
         
