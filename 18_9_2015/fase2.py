

import cv2
import time
import numpy as np

def HarrisCorners(img,height,width):
	
	gray = np.float32(img)
	dst = cv2.cornerHarris(gray,2,3,0.04)
	
	dst = cv2.dilate(dst,None)
	
	# Threshold for an optimal value, it may vary depending on the image.
	img[dst>0.02*dst.max()]=[255]
   
	# draw a small circle on the original image
	cv2.imshow('dst',img)
	
	#cv2.circle(image,[10,10],3,(0,255,0),-1)
	#cv2.circle(image,[120,120],3,(0,255,0),-1)
 
 

def Fase2(number,rect,resolutions,camera,r0,r1,r2,r3,fase1_resolution,et,Debug,usa_ottimizzazione_statica):
		
	# set the resolution for this fase
	w,h = resolutions[6] # '424.0', '240.0'
	fase2_resolution = set_res(camera,int(float(w)),int(float(h)))

	min_rect = r2/2 # start with an "empiric" minimal rect (based on width)
	old_end = 1.0
	old_number = number
	optimized = 0
	best_minrect_array = [0] * 500 # create an array of 500 values 

	while number<1000:
		start = time.time()
		(grabbed, image) = camera.read()
		# grab the raw NumPy array representing the image
		
		# check to see if we have reached the end of the video in case of video file
		if not grabbed:
			print "break 2"
			#SendToArduino("break 2")
			break
		#frame = image[r0:r2 , r1:r3]
		#frame = image[r0:r2 , r1:r3]
		tollerance = 50 # find a different way to calcolate this
		 
		moltiplicator_w = fase2_resolution[0] / fase1_resolution[0]
		moltiplicator_h = fase2_resolution[1] / fase1_resolution[1]
		
		# this is not perfect, shud check how to improve
		rr0 = int(int(r0 -tollerance) * moltiplicator_w)
		rr1 = int(int(r1 -tollerance) * moltiplicator_h)
		rr2 = int(int(r2 + tollerance) * moltiplicator_w)
		rr3 = int(int(r3 + tollerance) * moltiplicator_h)
		
		
		#print "ok",r0,r1,r2,r3
		#print "ok",rr0,rr1,rr2,rr3
		
		#break
	  
		frame = image[rr1:rr3 , rr0:rr2]
		# resize the frame and convert it to grayscale
		#frame = imutils.resize(frame, width = 300)
		gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
		gray = cv2.blur(gray, (3,3))
		
		# detect eyes in the image
		rects = et.track(gray,(min_rect,min_rect))
		
		# loop over the face bounding boxes and draw them
		for rect in rects:
			cv2.rectangle(frame, (rect[0], rect[1]), (rect[2], rect[3]), (0, 255, 0), 2)
			number += 1
			HarrisCorners(gray,rect[0], rect[1])
			if Debug:
				print number
			# save the current min_rect value into the array 
			best_minrect_array[min_rect] = int (best_minrect_array[min_rect]) +1
		
		# show the tracked eyes and face, then clear the
		# frame in preparation for the next frame
		if Debug:
			cv2.imshow("Eye Tracking", frame)
		#rawCapture.truncate(0)
		end = time.time()
		elapsed_time = end - start
		
		
		if usa_ottimizzazione_statica:
			# static optimization
			if (elapsed_time < old_end) & (old_number <> number) :
				if optimized < 100 :
					optimized +=1
					min_rect +=int(min_rect*5/100) 
					
			else:
				if optimized < 100:
					optimized +=1
					min_rect -=int(min_rect*5/100) 
			# after last update take some margin and fix the value of min_rect for future recognition    
			if optimized == 99:
				min_rect -= int(min_rect*5/100)
				optimized = 1000
		else:
			# dinamic optimization
			if (elapsed_time < old_end) & (old_number <> number) :
				min_rect +=3
			else:
				min_rect -=6
				
		if Debug:
			print elapsed_time,min_rect,rect[2]
		old_end = end
		old_number = number
		if min_rect < 10:
			print "min_rect too small "
			break
		if min_rect > 600:
			print "min_rect too large"
			break
		# if the 'q' key is pressed, stop the loop
		if cv2.waitKey(1) & 0xFF == ord("q"):
			break
			
	return min_rect,best_minrect_array,fase2_resolution
 



def set_res(cap, x,y):
    cap.set(cv2.cv.CV_CAP_PROP_FRAME_WIDTH, int(x))
    cap.set(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT, int(y))
    return float(cap.get(cv2.cv.CV_CAP_PROP_FRAME_WIDTH)),float(cap.get(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT))
    
