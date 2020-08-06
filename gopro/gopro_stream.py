import cv2
import socket
from goprocam import GoProCamera, constants
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from imutils.video import VideoStream
import numpy as np
import imutils
import time
import os

def detect_and_predict(frame, faceNet, maskNet):
    # GRAB DIMENSIONS AND CONSTRUCT BLOB
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1.0, (224, 224),
    (104.0, 177.0, 123.0))


    # PASS BLOB THROUGH NETWORK AND OBTAIN FACE DETECTION
    faceNet.setInput(blob)
    detections = faceNet.forward()
    print(detections.shape)


    # INITIALIZE LIST OF FACES, CORRESPONDING LOCATIONS AND LIST OF PREDICTIONS FROM FACE MASK NETWORK
    faces = []
    locations = []
    predictions = []


    # LOOP OVER DETECTIONS
    for i in range(0, detections.shape[2]):
        # EXTRACT PROBABILITY ASSOCIATED WITH DETECTION
        confidence = detections[0, 0, i, 2]

    # FILTER OUT WEAK DETECTION BY PROBABILITY > MINIMUM PROBABILITY REQUIRED
        if confidence > 0.5 :

            # COMPUTE (X, Y)-COORDINATES OF BOUNDING BOW FOR OBJECT
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (start_x, start_y, end_x, end_y) = box.astype("int")

            # ENSURE BOUNDING BOX FALL WITHIN DIMENSIONS OF FRAME
            (start_x, start_y) = (max(0, start_x), max(0, start_y))
            (end_x, end_y) = (min(w - 1, end_x), min(h - 1, end_y))

            # EXTRACT FACE ROI, CONVERT IT FROM BGR TO RGB CHANNEL ORDERING, RESIZE TO 224x224, PREPROCESS IT
            face = frame[start_y:end_y, start_x:end_x]
            face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            face = cv2.resize(face, (224, 224))
            face = img_to_array(face)
            face = preprocess_input(face)

            # ADD FACE AND BOUNDING BOXES TO RESPECTIVE LISTS
            faces.append(face)
            locations.append((start_x, start_y, end_x, end_y))
   
    # MAKE PREDICTION ONLY IF FACE DETECTED
    if len(faces) > 0 :

        # FOR FASTER PREDICTIONS --> PREDICTION ON  ALL FACES AT THE SAME TIME RATHER THAN ONE-BY-ONE
        faces = np.array(faces, dtype = "float32")
        predictions = maskNet.predict(faces, batch_size = 32)
    
    # RETURN A 2-TUPLE OF FACE LOCATION & CORRESPONDING PREDICTIONS
    return(locations, predictions)

# LOAD FACE DETECTOR MODEL FROM DISK
prototxt_path = r"detector/deploy.prototxt"
weights_path = r"detector/res10_300x300_ssd_iter_140000.caffemodel"
faceNet = cv2.dnn.readNet(prototxt_path, weights_path)

# LOAD FACE DETECTOR MODEL FROM DISK
maskNet = load_model("mask_detector.model")

WRITE = False
gpCam = GoProCamera.GoPro(constants.gpcontrol)
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
t=time()
gpCam.livestream("start")
gpCam.video_settings(res='1080p', fps='30')
gpCam.gpControlSet(constants.Stream.WINDOW_SIZE, constants.Stream.WindowSize.R720)
cap = cv2.VideoCapture("udp://10.5.5.9:8554", cv2.CAP_FFMPEG)
counter = 0
while True:
    nmat, frame = cap.read()
    cv2.imshow("GoPro OpenCV", frame)
    if WRITE == True:
        cv2.imwrite(str(counter)+".jpg", frame)
        counter += 1
        if counter >= 10:
            break
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    if time() - t >= 2.5:
        sock.sendto("_GPHD_:0:0:2:0.000000\n".encode(), ("10.5.5.9", 8554))
        t=time()
# When everything is done, release the capture
cap.release()
cv2.destroyAllWindows()