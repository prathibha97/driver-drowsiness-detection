import torch
from matplotlib import pyplot as plt
import numpy as np
import cv2
import uuid
import os
import time

#import yolov5 module from torch hub
# model = torch.hub.load('ultralytics/yolov5', 'yolov5s')

#---------Object Detection using image --------#

# img = 'https://www.iii.org/sites/default/files/p_cars_highway_522785736.jpg'

# #detect objects in the image and print the results
# results = model(img)
# results.print()

# #show the image with the detected objects
# results.show()

# #save the image with the detected objects
# plt.imshow(np.squeeze(results.render()))
# plt.show()

#---------Object Detection using realtime video --------#

# cap = cv2.VideoCapture(0)
# while cap.isOpened():
#     returnValue, frame = cap.read()

#     #detect objects in real time
#     results = model(frame)

#     cv2.imshow('Drowsiness Detection', np.squeeze(results.render()))
#     #waits for a key press with a delay of 10 milliseconds and then checks if the pressed key is 'q' (ASCII code 113). If the 'q' key is pressed, the loop will break, and the program will move to the next line.
#     if cv2.waitKey(10) & 0xFF == ord('q'):
#         break

# cap.release()
# cv2.destroyAllWindows()

#---------Load custom trained model --------#

model = torch.hub.load('ultralytics/yolov5', 'custom', path='yolov5/runs/train/exp/weights/best.pt', force_reload=True)
cap = cv2.VideoCapture(0)
while cap.isOpened():
    ret, frame = cap.read()
    
    # Make detections 
    results = model(frame)
    
    cv2.imshow('Drowsiness Detection', np.squeeze(results.render()))
    
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()

# https://universe.roboflow.com/hudzaifah-makoto-ffg32/eyes-nawse