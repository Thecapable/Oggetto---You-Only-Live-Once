#importing necessary packages
import cv2
from darkflow.net.build import TFNet
import matplotlib.pyplot as plt
import pyttsx3
import os
import sys
import numpy as np
def image(a):
    engine = pyttsx3.init()

    set1 = [] 

    options = {
        'model': 'cfg/yolo.cfg',
        'load': 'bin/yolov2.weights',
        'threshold': 0.4,  #threshold value decides above what confidence level the objects need to be detected
        #Higher Threshold equals Lesser chance of inaccurate results. 
        #You can change the threshold value to anywhere between 0-1 and see the changes it makes yourself
    }
    colors = [tuple(255 * np.random.rand(3)) for _ in range(10)]

    tfnet = TFNet(options)
    img = cv2.imread(a)

# use YOLO to predict the image
    results = tfnet.return_predict(img)
    engine.say('objects are')

    img.shape
    for color, result in zip(colors, results):
            tl = (result['topleft']['x'], result['topleft']['y'])
            br = (result['bottomright']['x'], result['bottomright']['y'])
            label = result['label']
            confidence = result['confidence']
            text = '{}: {:.0f}%'.format(label, confidence * 100)
            img = cv2.rectangle(img, tl, br, color, 3)  #for creating the bounding boxes around objects
            img = cv2.putText(img, label, tl, cv2.FONT_HERSHEY_COMPLEX,2, tuple(255 * np.random.rand(3)),2) 
            
            engine.say(label)
    
    engine.runAndWait()

# add the box and label and display it
 
    plt.imshow(img)
    plt.show()
