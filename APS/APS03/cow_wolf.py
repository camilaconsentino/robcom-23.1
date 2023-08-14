#!/usr/bin/python
# -*- coding: utf-8 -*-

import cv2
import numpy as np
from biblioteca_cow import *

cap = cv2.VideoCapture('cow_wolf/cow_wolf.mp4')

# Classes da MobileNet
CLASSES = [None]

while(True):
    # Capture frame-by-frame
    ret, img = cap.read()
    
    if ret == False:
        print("Codigo de retorno FALSO - problema para capturar o frame")
        break
    else:
        # Classes

        CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
            "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
            "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
            "sofa", "train", "tvmonitor"]
        net = load_mobilenet()
        CONFIDENCE = 0.7
        COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))
        
        image, resultados = detect(net, img, CONFIDENCE, COLORS, CLASSES)
        img, animais = separar_caixa_entre_animais(img, resultados)
        img = checar_perigo(img, animais)


        cv2.imshow('img',img)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()