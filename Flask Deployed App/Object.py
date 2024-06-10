# -*- coding: utf-8 -*-
"""
Created on Sun Apr 21 14:13:51 2024

@author: Rithik Raj
"""

import numpy as np
import imutils
from imutils.video import VideoStream, FPS
import cv2
import time
import argparse
 
classes = ['background', 'aeroplane', 'bicycle', 'bird', 'boat', 'bus', 'car',
           'cat', 'chair', 'dog', 'door', 'elephant', 'fish', 'fan', 'gun',
           'horse', 'knife', 'lion', 'motorbike', 'person', 'sheep', 'sofa',
           'train', 'tv', 'yatch']

colors = np.random.uniform(0, 255, size=(len(classes), 3))

print("Loading Model....")
net = cv2.dnn.readNetFromCaffe('MobileNetSSD_deploy.prototxt.txt', 
                               'MobileNetSSD_deploy.caffemodel')

print("Start Video Stream..")
vs = VideoStream(src=0).start()
time.sleep(2.0)
fps = FPS().start()

while True:
    frame = vs.read()
    frame = imutils.resize(frame, width=500)
    
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (400, 400)),
                                 0.005, (400, 400), 128)
    net.setInput(blob)
    detection = net.forward()
    
    for i in np.arange(0, detection.shape[2]):
        confidence = detection[0, 0, i, 2]
        if confidence > 0.25:
            idx = int(detection[0, 0, i, 1])
            box = detection[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startx, starty, endx, endy) = box.astype("int")
            label = "{}: {:.2f}%".format(classes[idx], confidence * 100)
            cv2.rectangle(frame, (startx, starty), (endx, endy), colors[idx], 2)
            y = starty - 15 if starty - 15 > 15 else starty + 15
            cv2.putText(frame, label, (startx, y),
				cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors[idx], 2)
    
    cv2.imshow('Output', frame)
    key = cv2.waitKey(10) & 0xFF
    if key == ord('q'):
        break
    
    fps.update()
       
fps.stop()
print("[INFO] elapsed time: {:.2f}".format(fps.elapsed()))
print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))

cv2.destroyAllWindows()
vs.stop()                    
                           
                           
                           
                           
                           
                           
                           