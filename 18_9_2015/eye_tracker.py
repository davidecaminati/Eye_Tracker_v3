# import the necessary packages
from pyimagesearch.eyetracker_no_face import Eyetracker_no_face
from pyimagesearch import imutils

#from skimage.filter import threshold_adaptive
import numpy as np

import argparse
import time
import cv2

import scipy.spatial

from collections import Counter
from fase1 import Fase1
from fase2 import Fase2
from fase3 import Fase3

import time

import random

import math

import socket

class Eye_tracker:
	
	# how many eye detection could be achieved every 50 frame (0.8 = 80%)
	minimal_quality = 0.01  
	# construct the eye tracker and allow the camera to worm up
	et = Eyetracker_no_face("cascades/haarcascade_eye.xml")
	#    list of possible resolution of my Logitech C270 camera
	resolutions = [('640.0', '480.0'), ('160.0', '120.0'), ('176.0', '144.0'), ('320.0', '176.0'), ('320.0', '240.0'), ('352.0', '288.0'), ('424.0', '240.0'), ('432.0', '240.0'), ('544.0', '288.0'), ('640.0', '360.0'), ('752.0', '416.0'), 
    ('800.0', '448.0'), ('800.0', '600.0'), ('856.0', '480.0'), ('864.0', '480.0'), ('960.0', '544.0'), ('960.0', '720.0'), ('1024.0', '576.0'), ('1184.0', '656.0'), ('1280.0', '960.0')]

	#camera = None
	number = 0
	#eye = None
	fase1_resolution = None
	fase2_resolution = None
	#debug = False
	rotation_angle = 0.0
	w = 0
	h = 0
	point_of_rotation = (1,1) #initial fake point
	eye_detected = 0
	
	def __init__(self,debug,static_optimization = True,eye = 0,video = None):
		self.debug = debug
		self.static_optimization = static_optimization
		self.eye = eye
		self.video = video
		
    
	def start(self):
		#find camera or video path
		video_src = self.video if (self.video != None) else self.findCamera()

		# Set camera for capture frames from the input
		self.camera = cv2.VideoCapture(video_src)
		
		# set the resolution for Fase1
		w,h = self.resolutions[6]
		self.fase1_resolution = self.set_camera_res(self.camera,w,h)
		print "inizio"
		
		number_eyes,rects = self.count_eye_detected(50,True)
		
		if number_eyes == 2:
			EyeROI = rects[0]
			print EyeROI
			# test FindCircle function
			self.FindCircle(EyeROI)
			# perform eye color detection
			self.FindColor()
			# perform crop of eye
			self.HarrisCorners()
			# run live camera
			'''
			while True:
				(grabbed,frame) = self.camera.read()
				if grabbed:
					img = self.rotate(frame,self.rotation_angle,center = self.point_of_rotation)
					cv2.imshow("LIVE", img)
					# if the 'q' key is pressed, stop the loop (Note: waitKey are necessary for display camera output)
					if cv2.waitKey(1) & 0xFF == ord("q"):
						break
				else:
					break
			'''
		else:
			print "errore nel riconoscimento occhi"
		
		
	def HarrisCorners(self):
		primo = 1
		while True:
			(grabbed, frame) = self.camera.read()
			gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
			gray = np.float32(gray)
			dst = cv2.cornerHarris(gray,2,3,0.02)
			
			dst = cv2.dilate(dst,None)
			
			# Threshold for an optimal value, it may vary depending on the image.
			frame[dst>0.02*dst.max()]=[255]
		   
			# draw a small circle on the original image
			cv2.imshow('dst',frame)
			
			#cv2.circle(image,[10,10],3,(0,255,0),-1)
			#cv2.circle(image,[120,120],3,(0,255,0),-1)
			
			# if the 'q' key is pressed, stop the loop (Note: waitKey are necessary for display camera output)
			if cv2.waitKey(1) & 0xFF == ord("q"):
				break
			elif cv2.waitKey(1) & 0xFF == ord("a"):
				primo = 1
			elif cv2.waitKey(1) & 0xFF == ord("s"):
				primo = 0


	def FindColor(self):
		primo = 1
		# define the list of boundaries
		boundaries = [
			([17, 15, 100], [50, 56, 200]),
			([86, 31, 4], [220, 88, 50])#,
			#([25, 146, 190], [62, 174, 250]),
			#([103, 86, 65], [145, 133, 128])
		]
			
		while True:
			(grabbed, frame) = self.camera.read()
			# define the list of boundaries
			for (lower, upper) in boundaries:
				# create NumPy arrays from the boundaries
				lower = np.array(lower, dtype = "uint8")
				upper = np.array(upper, dtype = "uint8")
			 
				# find the colors within the specified boundaries and apply
				# the mask
				mask = cv2.inRange(frame, lower, upper)
				output = cv2.bitwise_and(frame, frame, mask = mask)
			 
				# show the images
				cv2.imshow("images", np.hstack([frame, output]))
			if cv2.waitKey(1) & 0xFF == ord("q"):
				break
	
	def FindCircle(self,EyeROI):
		tollerance = 0
		r0 = EyeROI[0]
		r1 = EyeROI[1]
		r2 = EyeROI[2]
		r3 = EyeROI[3]
		# set the resolution for Fase2
		w,h = self.resolutions[0]
		self.fase2_resolution = self.set_camera_res(self.camera,w,h)
		
		moltiplicator_w = self.fase2_resolution[0] / self.fase1_resolution[0]
		moltiplicator_h = self.fase2_resolution[1] / self.fase1_resolution[1]
		
		rr0 = int(int(r0 - tollerance) * moltiplicator_w)
		rr1 = int(int(r1 - tollerance) * moltiplicator_h)
		rr2 = int(int(r2 + tollerance) * moltiplicator_w)
		rr3 = int(int(r3 + tollerance) * moltiplicator_h)
		
		
		while True:
			(grabbed, frame) = self.camera.read()
			# adapt the coordination for translate to the new camera resolution
			x,y = self.point_of_rotation
			new_point_of_rotation_x =int(int(x ) * moltiplicator_w)
			new_point_of_rotation_y =int(int(y ) * moltiplicator_h)
			# rotate the image
			frame = self.rotate(frame,self.rotation_angle,(new_point_of_rotation_x,new_point_of_rotation_y))
			# crop the image
			frame = frame[rr1:rr3 , rr0:rr2]

			output = frame.copy()
			gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
			# detect circles in the image
			circles = cv2.HoughCircles(gray, cv2.cv.CV_HOUGH_GRADIENT, 2.5, 100,minRadius=20,maxRadius=30)
			 
			# ensure at least some circles were found
			if circles is not None:
				# convert the (x, y) coordinates and radius of the circles to integers
				circles = np.round(circles[0, :]).astype("int")
			 
				# loop over the (x, y) coordinates and radius of the circles
				for (x, y, r) in circles:
					# draw the circle in the output image, then draw a rectangle
					# corresponding to the center of the circle
					cv2.circle(output, (x, y), r, (0, 255, 0), 4)
					#cv2.rectangle(output, (x - 5, y - 5), (x + 5, y + 5), (0, 128, 255), -1)
					pupil = output[x-2: y - 2, x + 2: y + 2]
					#cv2.imshow("pupil", pupil)
					#c = FindPredominantColor(x,y,r)
				# show the output image
				cv2.imshow("output", output)
				#cv2.waitKey(0)
			cv2.imshow("occhio", frame)
			if cv2.waitKey(1) & 0xFF == ord("q"):
				break
				
	# check if you sill using an UDOO Board
	def findCamera(self):
		# check if we running on UDOO board ( in udoo the USB webcam is /dev/video3 )
		name = socket.gethostname()
		if name == 'udoobuntu':
			return 3
		else:
			return 0
	
	def FindPredominantColor(self,x,y,r):
		pass
			
	def find_resolution_for_camera(self):
		# use this to find resolution available take 10 minutes to run
		valArray = []
		for numx in range(100,1300,10):  #to iterate between 10 to 1300 step 10
			for numy in range(100,1300,10):  #to iterate between 10 to 1300 step 10
				print numx,numy
				val = self.set_camera_res(self.camera,numx,numy)
				if val not in valArray:
					valArray.append(val)
		return valArray

		
	def set_camera_res(self,cap, x,y):
		cap.set(cv2.cv.CV_CAP_PROP_FRAME_WIDTH, int(float(x)))
		cap.set(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT, int(float(y)))
		return float(cap.get(cv2.cv.CV_CAP_PROP_FRAME_WIDTH)),float(cap.get(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT))
		
		
	def count_eye_detected(self,number_frame_to_check,show_rectangle = True,accuracy = 0.5):
		# if frames with 1 eye detected are 2 times more than frames with 2 eyes detected, return 1 otherwise return 2
		# if total frame with eye detected are less then 50% (default accuracy) return 0
		_accuracy = accuracy
		_frame_number = 0
		_number_frame_1_eye = 0.0
		_number_frame_2_eye = 0.0
		_number_frame_to_check = number_frame_to_check
		_show_rectangle = show_rectangle

		
		while _frame_number < _number_frame_to_check:
			#print _frame_number
			(grabbed, frame) = self.camera.read()		
			# check to see if we have reached the end of the
			# video
			if not grabbed:
				print "no grab"
				break
			
			# resize the frame and convert it to grayscale
			#frame = imutils.resize(frame, width = 300)
			gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
			gray = cv2.blur(gray, (3,3))
			if (_frame_number % 60*30*10) == 0 or _frame_number == 0:
				error, rotation = self.adjust_image_rotation(10)
				if error:
					return 0
			'''
			# test rotation with the first 10 frames
			if i <= 10:  
				error,rotation = self.adjust_image_rotation(gray)
				#rotations.add(rotation)
				if error == 1:
					print "errore, non sono visibili entrambi gli occhi"
					#break
				if error == 0:
					mydata[i] = rotation
					i +=1
			
			print "mydata" , mydata
			#time.sleep(1)
			if i == 10:
				res = self.kmeans(mydata,k=2)
				print "res" , res
				time.sleep(10)
			'''			
			gray = self.rotate(gray,self.rotation_angle,self.point_of_rotation)
							
			# detect eyes in the image
			rects = self.et.track(gray)
			_frame_number += 1
			if len(rects) == 1:
				_number_frame_1_eye += 1
			if len(rects) == 2:
				_number_frame_2_eye += 1
					
			if _show_rectangle:
				# loop over the eyes bounding boxes and draw them
				for rect in rects:
					(h, w) = frame.shape[:2]
					if self.debug:
						print h, w
						pass
					cv2.rectangle(gray, (rect[0], rect[1]), (rect[2], rect[3]), (0, 255, 0), 2)
					
			cv2.imshow("Tracking", gray)
			# if the 'q' key is pressed, stop the loop (Note: waitKey are necessary for display camera output)
			if cv2.waitKey(1) & 0xFF == ord("q"):
				break
				
		if ((_number_frame_1_eye + _number_frame_2_eye) / number_frame_to_check) < _accuracy:
			return 0
		else:
			if _number_frame_1_eye /2 >  _number_frame_2_eye:
				return 1,rects
			else:
				return 2,rects
				
	def adjust_image_rotation(self,number_frame_to_check):
		rotation = 0.0
		_min_deviance_for_eyes = 10 # how much pixel of different height we could have between 2 eyes
		_max_degree_of_difference = 3
		_frame_number = 0
		_number_frame_to_check = number_frame_to_check	
			
		rotation_angles = np.zeros((_number_frame_to_check+1, 1))
		self.point_of_rotation = (self.w/2,self.h/2)
		
		print "rotation_angles" , rotation_angles
		i = 0
		#time.sleep(3)
		error = False
		while i <= _number_frame_to_check:
			(grabbed, frame) = self.camera.read()
			w,h = frame.shape[:2]
			self.point_of_rotation = (h/2,w/2)
			print "w,h" ,w,h
			#time.sleep(2)
			if not grabbed:
				print "no grab"
				break
			gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
			gray = cv2.blur(gray, (3,3))
			
			'''
			error,rotation = self.adjust_image_rotation(gray,10)
			#rotations.add(rotation)
			if error == 1:
				print "errore, non sono visibili entrambi gli occhi"
				#break
			if error == 0:
				rotation_angles[i] = rotation
				i +=1
		
			print "rotation_angles" , rotation_angles
			#time.sleep(1)
			if i == 10:
				res = self.kmeans(rotation_angles,k=2)
				print "res" , res
				time.sleep(10)
			'''
			
			error = False
			y1 = 0
			y2 = -1000
			_max_fail = 10 # max fail in detect 2 eyes
			_max_iteration = 50
			_iteraction = 0
			while abs(y1 - y2) > _min_deviance_for_eyes:
				
				_iteraction += 1
				if _iteraction > _max_iteration:
					print "antiloop"
					time.sleep(3)
					error = True
					break
				img = frame
				img = self.rotate(img,rotation,center = self.point_of_rotation)
					
				rects = self.et.track(img)
				#cv2.rectangle(img, self.point_of_rotation,self.point_of_rotation, (255), 20)
				
				cv2.imshow("rotated", img)	
				if len(rects) == 2:
					#self.point_of_rotation = ((rects[0][0] +rects[0][2])/2,(rects[0][1]+rects[0][3])/2)
					
					y1 = (rects[0][3] - rects[0][1]) /2 + rects[0][1]
					x1 = (rects[0][2] - rects[0][0]) /2 + rects[0][0]
					
					y2 = (rects[1][3] - rects[1][1]) /2 + rects[1][1]
					x2 = (rects[1][2] - rects[1][0]) /2 + rects[1][0]
					
					x_left = x1 if (x1 < x2) else x2
					y_left = y1 if (x1 < x2) else y2
					
					x_right = x2 if (x1 < x2) else x1
					y_right = y2 if (x1 < x2) else y1
					
					#rotation -= 0.3 +round(random.random(), 2)
					
					if y_left > y_right:
						rotation -= 0.3 +round(random.random(), 2)
					if y_left < y_right:
						rotation += 0.3 +round(random.random(), 2)
					
					print rotation
					if y_left == y_right:
						break
					# if the 'q' key is pressed, stop the loop (Note: waitKey are necessary for display camera output)
					if cv2.waitKey(1) & 0xFF == ord("q"):
						break
				
				if len(rects) != 2:
					print "occhi non rilevati" 
					rotation = random.randrange(-30,30)
					_max_fail -= 1
				if _max_fail == 0:
					error = True
					break 
				
			if not error :
				rotation_angles[i] = rotation
				print "i = " , i
				#time.sleep(2)
				i +=1
			
		#if there are 2 rotation too (_max_degree_of_difference) different , remove it
		_max = np.amax(rotation_angles)
		_min = np.amin(rotation_angles)

		while abs(_max -_min) > _max_degree_of_difference :
			_max = np.amax(rotation_angles)
			_min = np.amin(rotation_angles)
			if len(rotation_angles) > 2:
				rotation_angles = np.delete(rotation_angles,np.where(rotation_angles==_max)[0][0])
				rotation_angles = np.delete(rotation_angles,np.where(rotation_angles==_min)[0][0])
			#print "removed"
			#print rotation_angles
			#time.sleep(1)
		
		if len(rotation_angles) >= 2:
			self.rotation_angle = np.mean(rotation_angles)
			#print self.rotation_angle 
			#time.sleep(10)
		else:
			error = True
			return error,rotation
		'''
		print rotation_angles
		print "remove sequence finisched"
		time.sleep(10)
		
		res = self.kmeans(rotation_angles,k=3)
		print "res" , res
		res = self.kmeans(rotation_angles,k=3)
		print "res" , res
		res = self.kmeans(rotation_angles,k=3)
		print "res" , res
		print "rotation_angles" , rotation_angles
		time.sleep(10)
		'''
		return error,rotation

		
	def rotate(self, image, angle, center = None, scale = 1.0):
		_image = image
		(h, w) = _image.shape[:2]
		
		if center is None:
			center = (w / 2, h / 2)
		
		M = cv2.getRotationMatrix2D(center, angle, scale)
		rotated = cv2.warpAffine(_image, M, (w, h))
		
		return rotated
			
		
	def localize_eyes_in_face(self):
		
		rectArray = []
		number_common_rect = 0
		b = Counter(rectArray)
		while number_common_rect < 10: #at least 3 entry of the same rect

			start = time.time()
			
			#find the must common rect
			if len(rectArray) > 5:
				b = Counter(rectArray)
				if self.debug:
					print "b.most_common(1)" + str(b.most_common(1))
					print "number = " + str(b.most_common(1)[0][1])
				number_common_rect = b.most_common(1)[0][1]
				
			(grabbed, frame) = self.camera.read()
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
			rects = self.et.track(gray)
			
			# loop over the eyes bounding boxes and draw them
			for rect in rects:
				(h, w) = frame.shape[:2]
				if self.debug:
					print rect[0],h,w
				cv2.rectangle(frame, (rect[0], rect[1]), (rect[2], rect[3]), (0, 255, 0), 2)
				r0 = rect[0]
				r1 = rect[1]
				r2 = rect[2]
				r3 = rect[3]
				if self.eye == "0": # Left eye
					if rect[0] <= w/2:
						self.number += 1
						if self.debug:
							print "left"
						rectArray.append(rect)
				else:                   # Right eye
					if rect[0] >= w/2:
						self.number += 1
						if self.debug:
							print "right"
						rectArray.append(rect)
				
			
			# show the tracked eyes 
			if self.debug:
				cv2.imshow("Tracking", frame)
			# clear the frame in preparation for the next frame
			#rawCapture.truncate(0)
			
			# calcolate performance
			end = time.time()
			difference = end - start
			
			if self.debug:
				print difference
			##SendToArduino(difference)
			
			# if the 'q' key is pressed, stop the loop
			if cv2.waitKey(1) & 0xFF == ord("q"):
				break
		return rectArray,rect,b
			
				
	def cluster_centroids(self,mydata, clusters, k=None):
		"""Return centroids of clusters in data.

		data is an array of observations with shape (A, B, ...).

		clusters is an array of integers of shape (A,) giving the index
		(from 0 to k-1) of the cluster to which each observation belongs.
		The clusters must all be non-empty.

		k is the number of clusters. If omitted, it is deduced from the
		values in the clusters array.

		The result is an array of shape (k, B, ...) containing the
		centroid of each cluster.

		>>> data = np.array([[12, 10, 87],
		...                  [ 2, 12, 33],
		...                  [68, 31, 32],
		...                  [88, 13, 66],
		...                  [79, 40, 89],
		...                  [ 1, 77, 12]])
		>>> cluster_centroids(data, np.array([1, 1, 2, 2, 0, 1]))
		array([[ 79.,  40.,  89.],
			   [  5.,  33.,  44.],
			   [ 78.,  22.,  49.]])

		"""
		if k is None:
			k = np.max(clusters) + 1
		result = np.empty(shape=(k,) + mydata.shape[1:])
		for i in range(k):
			np.mean(mydata[clusters == i], axis=0, out=result[i])
		return result


	def kmeans(self,mydata, k=None, centroids=None, steps=200):
		"""Divide the observations in data into clusters using the k-means
		algorithm, and return an array of integers assigning each data
		point to one of the clusters.

		centroids, if supplied, must be an array giving the initial
		position of the centroids of each cluster.

		If centroids is omitted, the number k gives the number of clusters
		and the initial positions of the centroids are selected randomly
		from the data.

		The k-means algorithm adjusts the centroids iteratively for the
		given number of steps, or until no further progress can be made.

		>>> data = np.array([[12, 10, 87],
		...                  [ 2, 12, 33],
		...                  [68, 31, 32],
		...                  [88, 13, 66],
		...                  [79, 40, 89],
		...                  [ 1, 77, 12]])
		>>> np.random.seed(73)
		>>> kmeans(data, k=3)
		array([1, 1, 2, 2, 0, 1])

		"""
		if centroids is not None and k is not None:
			assert(k == len(centroids))
		elif centroids is not None:
			k = len(centroids)
		elif k is not None:
			# Forgy initialization method: choose k data points randomly.
			centroids = mydata[np.random.choice(np.arange(len(mydata)), k, False)]
		else:
			raise RuntimeError("Need a value for k or centroids.")

		for _ in range(max(steps, 1)):
			# Squared distances between each point and each centroid.
			sqdists = scipy.spatial.distance.cdist(centroids, mydata, 'sqeuclidean')

			# Index of the closest centroid to each data point.
			clusters = np.argmin(sqdists, axis=0)

			new_centroids = self.cluster_centroids(mydata, clusters, k)
			if np.array_equal(new_centroids, centroids):
				break

			centroids = new_centroids

		return clusters
