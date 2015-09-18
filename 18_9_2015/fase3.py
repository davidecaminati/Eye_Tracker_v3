import time
import cv2
import serial
# pip install pyserial

time.sleep(2)



def Fase3(min_rect,fase2_resolution,best_minrect_array,Debug,fase1_resolution,r0,r1,r2,r3,camera,et,minimal_quality):

	# Serial comunication
	serPort = "/dev/ttyACM0"
	serPort2 = "/dev/ttyACM1"
	baudRate = 9600
	ser = serial
	ser2 = serial
	try:
		ser = serial.Serial(serPort, baudRate)
	except:
		pass
		
	try:
		ser2 = serial.Serial(serPort2, baudRate)
	except:
		pass
		
	performance_test_eyes = []*5
	total_frame_number = 0.0
	time_for_gesture = 40 # number of frames to check for the gesture
	gestureArray = ["C"] * time_for_gesture

	#print best_minrect_array
	number_of_good_min_rect = max(best_minrect_array)
	#print number_of_good_min_rect
	best_min_rect = best_minrect_array.index(number_of_good_min_rect)
	#print best_min_rect

	if number_of_good_min_rect > 1:
		#now i have a good reason to use best_min_rect as my min_rect
		if Debug:
			print "fase 3 started"
		SendToArduino("fase 3 started",ser,ser2)
		
		#setting of all the variable
		#min_rect = int(min_rect/1.3) # this si the key of speed and stability (1.3 seems good enought)
		tollerance = 60 # find a different way to calcolate this
		moltiplicator_w = fase2_resolution[0] / fase1_resolution[0]
		moltiplicator_h = fase2_resolution[1] / fase1_resolution[1]
		
		rr0 = int(int(r0 - tollerance) * moltiplicator_w)
		rr1 = int(int(r1 - tollerance) * moltiplicator_h)
		rr2 = int(int(r2 + tollerance) * moltiplicator_w)
		rr3 = int(int(r3 + tollerance) * moltiplicator_h)
		

		eye_frames = 0.0
		partial_frame_number = 0.0
		quality = 0.0
		consecutive_fail = [0]*300
		fail = 0
		elapsed_time = 0
		while True:
			
			start = time.time()
			
			partial_frame_number +=1 #increment every new frame 
			total_frame_number += 1.0
			(grabbed, image) = camera.read()
			
			# check to see if we have reached the end of the video in case of video file
			if not grabbed:
				print "end of video stream"
				SendToArduino("end of video stream",ser,ser)
				break

			frame = image[rr1:rr3 , rr0:rr2]
			# resize the frame and convert it to grayscale
			#frame = imutils.resize(frame, width = 300)
			gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
			gray = cv2.blur(gray, (3,3))
			# detect eyes in the image
			rects = et.track(gray,(min_rect,min_rect))
			if len(rects) == 0:
				fail +=1
			# loop over the face bounding boxes and draw them
			for rect in rects:
				cv2.rectangle(frame, (rect[0], rect[1]), (rect[2], rect[3]), (0, 255, 0), 2)
				eye_frames +=1
				if fail <> 0:
					consecutive_fail[int(total_frame_number)] = fail
				fail = 0
				#roi_eye = image[rr1:rr3 , rr0:rr2]
				#print rr0,rr1,rr2,rr3
				miox = (rect[1]+rect[3])/2
				roi_eye = frame[miox-20:miox+20,rect[0]:rect[2]]
				if Debug:
					cv2.imshow("roi_eye", roi_eye)
				
				
				#image2 = roi_eye.copy()
				
				# apply a Gaussian blur to the image then find the brightest
				# region
				roi_eye = cv2.cvtColor(roi_eye, cv2.COLOR_BGR2GRAY)
				roi_eye = cv2.equalizeHist(roi_eye)
				roi_eye = cv2.GaussianBlur(roi_eye,(5, 5), 0)
				(minVal, maxVal, minLoc, maxLoc) = cv2.minMaxLoc(roi_eye)
				#cv2.circle(image2, maxLoc, 5, (255, 0, 0), 2)
				x,y = minLoc
				#print x,y
				colore = (0,0,0)
				(h, w) = roi_eye.shape[:2]
				actual_gesture = 0
				eye_information = []*3
				if x <= (w/5)*2: # Left
					colore = (0,0,255)
					print "L"
					performance_test_eyes.append([elapsed_time,"L",h,w,total_frame_number])
					actual_gesture = GestureDetect(GestureEngine("L",gestureArray))
				elif x <= (w/5)*3: # Center
					colore = (255,255,255)
					print "C"
					
					performance_test_eyes.append([elapsed_time,"C",h,w,total_frame_number])
					actual_gesture =  GestureDetect(GestureEngine("C",gestureArray))
				else: # Right
					colore = (255,0,0)
					print "R"
					performance_test_eyes.append([elapsed_time,"R",h,w,total_frame_number])
					actual_gesture =  GestureDetect(GestureEngine("R",gestureArray))
				#print actual_gesture
				if actual_gesture == 1:
					SendToArduino("--1",ser,ser)
				if actual_gesture == 2:
					SendToArduino("--2",ser,ser)
				if actual_gesture == 3:
					SendToArduino("--L",ser,ser)
				if actual_gesture == 4:
					SendToArduino("--R",ser,ser)
				
				
				#cv2.circle(image2, minLoc, 5, colore, 2)
				#cv2.imshow("image2", image2)
				
				# display the results of our newly improved method
				#cv2.imshow("Robust", roi_eye)
				
				#r_eye = cv2.Canny(roi_eye,50,100)
					
				#test
				#lap = cv2.Laplacian(frame, cv2.CV_64F)
				#lap = np.uint8(np.absolute(lap))
				#cv2.imshow("Laplacian", lap)
				
				#time.sleep(0.1)
				#print "eye located"

			# show the tracked eyes
			#cv2.imshow("Eye Tracking", frame)
			#rawCapture.truncate(0)
			end = time.time()
			elapsed_time = end - start
			#print elapsed_time
			#SendToArduino(elapsed_time)
			
			# todo
			# find where you still looking (right, left, center)

			
			
			#quality test
			if partial_frame_number > 50:
				print "check!"
				if eye_frames > 1.0:
					quality = eye_frames/partial_frame_number
					eye_frames = 0.0
					partial_frame_number = 0.0
					if quality < minimal_quality:
						print "accuracy is too low", quality
						SendToArduino("accuracy is too low",ser,ser)
						#break
				else:
					print "the accuracy is too low no frame in last check!", quality
					SendToArduino("accuracy low no frame",ser,ser)
					#break
					
			# if the 'q' key is pressed, stop the loop
			if cv2.waitKey(1) & 0xFF == ord("q"):
				break
	else:
		text_to_print = "not enought min_rect", number_of_good_min_rect
		print text_to_print
		SendToArduino(text_to_print,ser,ser)
		ser.close
	return total_frame_number,performance_test_eyes,consecutive_fail
 
 
 
		
		
def SendToArduino(tts,ser,ser2):

	try:
		ser.write(str(tts))
		ser.write('\n') 
	except:
		pass
		
	try:
		ser2.write(str(tts))
		ser2.write('\n') 
	except:
		pass
    

def GestureDetect(gnd):
    
    green = ["C","L","C","R","C"]
    red = ["L","C","L","C"]
    #green = ["C","L","C","L"]
    
    #red = ["C","R","C","L","C"]
    left = ["C","R","C","R"]
    right = ["C","R","C","R","C","R","C"]
    
    #occhi = ["C","R","C","L","C","L","C","R","C","L"]
    
    
    if gnd == green:
        return 1
    elif gnd == red:
        return 2
    elif gnd == left:
        return 3
    elif gnd == right:
        return 4
    else:
        return 0
    

def GestureEngine(position,gestureArray):
    del gestureArray[0] # remove the oldest eye position
    gestureArray.append(position) # add the new eye position
    
    #remove duplicate contigue
    old_char = "N"
    gestureNoDuplicate = []
    for x in gestureArray:
        if x <> old_char:
            gestureNoDuplicate.append(x)
        old_char = x
    return gestureNoDuplicate
