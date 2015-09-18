# Detection of simple eye movement using Raspberry PI 2  or UDOO Quad and a cheap webcam
# developed based on example by Adrian Rosebrock (http://www.pyimagesearch.com/author/adrian/)
# writed by Davide Caminati 07/11/2015 (http://caminatidavide.it/)
# License GNU GPLv2

# USAGE
# python external_eyetraking_webcam.py -o True -e 0

# NOTE
# static_optimization parameter (-o) options:
# True fast performance, 
# False if the subject should change the distance of eye to the webcam during initial calibration (fase 1 and 2)


# import the necessary packages
from pyimagesearch.eyetracker_no_face import Eyetracker_no_face
from pyimagesearch import imutils

#from skimage.filter import threshold_adaptive
import numpy as np

import argparse
import time
import cv2
from collections import Counter
from fase1 import Fase1
from fase2 import Fase2, set_res
from fase3 import Fase3

import time

import socket


# TODO 
# read the camera resolution capability and save into an array (actually on test)
# Fase1 must identify correctly the eye position (no false positive) and select the eye to track
# add recognition procedute (hug, nn, whatelse)
# think about a routine to rotate image accordly with the face (or eye) rotation
# provide change of resolution of fase 1 and fase 2 as parameter (think on this)
# catch exception (eye not found)
# check if the eye is roughly in the center of the cam during Fase1
# auto find serial port for Arduino
# rotate image if necessary (test how much CPU consumer is and if it improve recognition)

# CAMERA NOTE
# i've tested some different camera for this software, that's my opinion:
# Microsoft LifeCam HD-3000 = good light, but slow during acquisition, difficult to hack lens
# LOGITECH HD C525 = very slow, wide lens make difficult to point a little element as eye, difficult to hack lens
# Logitech PC Webcam C270 = very cheaper, but easy to hack lens (you can easly remove the lens and replace it), very fast data throughput
# PS3 eye = very fast (it work on USB 2 and USB 3, manual focus with 2 preset (not enought for our scope) , very cheaper, good framerate but image not very clear, probalby you need a low pass filter to smooth the image


# --- the code start here ---



Debug = True

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video", required = False,
	help = "path to the video file, camera or raspicam")
ap.add_argument("-o", "--static_optimization", required = True,
	help = "True if you want static optimization")
ap.add_argument("-e", "--eye", required = True,
	help = "0 = Left or 1 = Right eye")
args = vars(ap.parse_args())

video_source = args["video"]
usa_ottimizzazione_statica = (args["static_optimization"] == "True")
eye_to_track = args["eye"] 


#set as default video source as webcam (suppose to use UDOO)
video_src = 3

# check if you sill using an UDOO Board
def findCamera():
    # check if we running on UDOO board ( in udoo the USB webcam is /dev/video3 )
    name = socket.gethostname()
    if name == 'udoobuntu':
        return 3
    else:
        return 0
        
# how many eye detection could be achieved every 50 frame (0.8 = 80%)
minimal_quality = 0.01  

# find video source (TODO)
if video_source == "raspicam":
    #use Raspicam as video grabber
    video_src = 0
elif video_source == "camera": 
    # default setting
    video_src = findCamera()
elif video_source is not None : 
    # load the video file
    video_src = video_source

print "video_src " + str(video_src) 
# initialize the camera and grab a reference to the raw camera capture

#camera = PiCamera()
#camera.resolution = (640, 480)
#camera.framerate = 32
#rawCapture = PiRGBArray(camera, size=(640, 480))

# construct the eye tracker and allow the camera to worm up
et = Eyetracker_no_face("cascades/haarcascade_eye.xml")
time.sleep(0.1)

# capture frames from the webcam
camera = cv2.VideoCapture(video_src)

#    list of possible resolution of my Logitech C270 camera
resolutions = [('640.0', '480.0'), ('160.0', '120.0'), ('176.0', '144.0'), ('320.0', '176.0'), ('320.0', '240.0'), ('352.0', '288.0'), ('424.0', '240.0'), ('432.0', '240.0'), ('544.0', '288.0'), ('640.0', '360.0'), ('752.0', '416.0'), 
    ('800.0', '448.0'), ('800.0', '600.0'), ('856.0', '480.0'), ('864.0', '480.0'), ('960.0', '544.0'), ('960.0', '720.0'), ('1024.0', '576.0'), ('1184.0', '656.0'), ('1280.0', '960.0')]

# use this to find resolution available take 10 minutes to run
'''
valArray = []
for numx in range(100,1300,10):  #to iterate between 10 to 1300 step 10
    for numy in range(100,1300,10):  #to iterate between 10 to 1300 step 10
        print numx,numy
        val = set_res(camera,numx,numy)
        if val not in valArray:
            valArray.append(val)
print valArray
'''
    
# set the resolution for this fase
w,h = resolutions[6]
print w,h
#time.sleep(100)
fase1_resolution = set_res(camera,int(float(w)),int(float(h)))

# debug
print "fase 1 started"
#SendToArduino("fase 1 started")

rectArray,number,rect,b = Fase1(camera,et,Debug,eye_to_track)

# use the must common rect finded in Fase1

rect = b.most_common(1)[0][0]
r0 = rect[0]
r1 = rect[1]
r2 = rect[2]
r3 = rect[3]


if Debug:
    #print "ok",r0,r1,r2,r3
    #time.sleep(10)
    
    # debug
    #time.sleep(10)
    print "fase 1 ended"
    print rectArray
#SendToArduino("fase 1 ended")
if Debug:
    print "fase 2 started"
#SendToArduino("fase 2 started")

min_rect,best_minrect_array,fase2_resolution = Fase2(number,rect,resolutions,camera,r0,r1,r2,r3,fase1_resolution,et,Debug,usa_ottimizzazione_statica)

print "fase 2 ended"

#SendToArduino("fase 2 ended")
# release resource 
#best_minrect_array = []

total_frame_number,performance_test_eyes,consecutive_fail = Fase3(min_rect,fase2_resolution,best_minrect_array,Debug,fase1_resolution,r0,r1,r2,r3,camera,et,minimal_quality)

#print performance_test_eyes
print "total_frame_number"
print total_frame_number
print "len(performance_test_eyes)"
valore = float(len(performance_test_eyes))
print valore
print "% of recognition"
print  float(float( 100.0 / total_frame_number) * valore)
aa = 0.0
for a,b,c,d,e in performance_test_eyes:
    aa += a
print float(aa /len(performance_test_eyes))
#print "consecutive_fail"
#print consecutive_fail
print "max(consecutive_fail)"
print max(consecutive_fail)



camera.release()
cv2.destroyAllWindows()
