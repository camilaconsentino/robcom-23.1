import cv2
import auxiliar as aux
import numpy as np
def colorpicker(frame, valor):
    img_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    hsv1, hsv2 = aux.ranges() #Alterar esse valor usando colorpicker
    mask = cv2.inRange(img_hsv, hsv1, hsv2)
    elemento_estrut = np.ones([3,3])
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, elemento_estrut)
    contornos, arvore = cv2.findContours(mask.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    #cv2.drawContours(frame, contornos, -1, [0,255,0], 3)
    return mask, contornos

# MASK VERDE
"#01ff02"

mask_verde, contornos = colorpicker(frame, "#01ff02")