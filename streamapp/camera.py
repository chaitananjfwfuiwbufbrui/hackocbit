# from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
# from tensorflow.keras.preprocessing.image import img_to_array
# from tensorflow.keras.models import load_model
from scipy.spatial import distance
import dlib
from playsound import playsound
from imutils import face_utils
from imutils.video import VideoStream
import imutils
from .models import *
import _thread
from media.song import *
import cv2
import os
import urllib.request
import numpy as np
from django.conf import settings
import time
face_detection_videocam = cv2.CascadeClassifier(os.path.join(
    settings.BASE_DIR, 'opencv_haarcascade_data/haarcascade_frontalface_default.xml'))
face_detection_webcam = cv2.CascadeClassifier(os.path.join(
    settings.BASE_DIR, 'opencv_haarcascade_data/haarcascade_frontalface_default.xml'))
# load our serialized face detector model from disk
prototxtPath = os.path.sep.join(
    [settings.BASE_DIR, "face_detector/deploy.prototxt"])
weightsPath = os.path.sep.join(
    [settings.BASE_DIR, "face_detector/res10_300x300_ssd_iter_140000.caffemodel"])
faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)
# maskNet = load_model(os.path.join(settings.BASE_DIR,'face_detector/mask_detector.model'))


def eye_aspect_ratio(eye):
    A = distance.euclidean(eye[1], eye[5])
    B = distance.euclidean(eye[2], eye[4])
    C = distance.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear


thresh = 0.25
frame_check = 20
detect = dlib.get_frontal_face_detector()
# Dat file is the crux of the code
predict = dlib.shape_predictor("models/shape_predictor_68_face_landmarks.dat")
(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS["right_eye"]


def Masking(frame):
    cv2.putText(frame, "****************ALERT!****************", (4, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    cv2.putText(frame, "****************ALERT!****************", (4, 265),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
	
	
def add_no():
    sa = Contract.objects.filter(user = "hello").last()

    sa.no_of_blinks = (sa.no_of_blinks%10)+ 1
    sa.save()
    

class VideoCamera(object):
    flag = 0

    def __init__(self):
        self.video = cv2.VideoCapture(0)

    def __del__(self):
        self.video.release()

    def get_frame(self):
        success, image = self.video.read()

        success, image = self.video.read()
        frame = image
        try:
            frame = imutils.resize(image, width=480)
        except:
            print("")
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        subjects = detect(gray, 0)
        for subject in subjects:
            global flag
            shape = predict(gray, subject)
            shape = face_utils.shape_to_np(shape)  # converting to NumPy Array
            leftEye = shape[lStart:lEnd]
            rightEye = shape[rStart:rEnd]
            leftEAR = eye_aspect_ratio(leftEye)
            rightEAR = eye_aspect_ratio(rightEye)
            ear = (leftEAR + rightEAR) / 2.0
            leftEyeHull = cv2.convexHull(leftEye)
            rightEyeHull = cv2.convexHull(rightEye)
            cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
            cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)
            if ear < thresh:
                flag += 1
                print(flag)

                if flag >= frame_check:
					
                    print("Drowsy1")
                    
					
                    # Buzzersound()
                    _thread.start_new_thread(Buzzersound, ())
                    _thread.start_new_thread(add_no, ())
                    # _thread.start_new_thread(Masking(frame ))
                    Masking(frame)
					

            else:
                flag = 0
        ret, jpeg = cv2.imencode('.jpg', frame)
        return jpeg.tobytes()
