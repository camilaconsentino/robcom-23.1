#clicar no quadradinho com a lupinha para vizualizar o arquivo md

import cv2
import numpy as np

#COLOR SEGMENTATION
def color_segmentation_star(img, lower, upper):
    hsv = cv2.cvtColor(img.copy(), cv2.COLOR_BGR2HSV)

    lower_hsv = np.array([lower//2,50,50],dtype=np.uint8) #yellow
    upper_hsv = np.array([upper//2,255,255],dtype=np.uint8)
    kernel = np.ones((5,5),np.uint8)

    mask = cv2.inRange(hsv, lower_hsv, upper_hsv)
    mask = cv2.morphologyEx(mask,cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask,cv2.MORPH_CLOSE, kernel)

    # find contours
    contours,_ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    if len(contours) > 0:
        # maior contorno (maior area)
        cnt = max(contours, key = lambda x: cv2.contourArea(x))

        # centro
        M = cv2.moments(cnt)
        centroX = int(M['m10']/M['m00'])
        centroY = int(M['m01']/M['m00'])
        point = [centroX, centroY]
        

    else:
        centroX = 0
        centroY = 0
        cnt = 0
        point = [centroX, centroY]     

    #cv2.imshow('MaskYellow', mask)
    #cv2.waitKey(1)   

    return point

#COLOR SEGMENTATION
def color_segmentation_triangle(img, lower, upper):
    hsv = cv2.cvtColor(img.copy(), cv2.COLOR_BGR2HSV)

    lower_hsv = np.array([lower//2,50,50],dtype=np.uint8) #yellow
    upper_hsv = np.array([upper//2,255,255],dtype=np.uint8)
    kernel = np.ones((5,5),np.uint8)

    mask = cv2.inRange(hsv, lower_hsv, upper_hsv)

    # find contours
    contours,_ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    if len(contours) > 0:
        # maior contorno (maior area)
        cnt = max(contours, key = lambda x: cv2.contourArea(x))

        # centro
        M = cv2.moments(cnt)
        centroX = int(M['m10']/M['m00'])
        centroY = int(M['m01']/M['m00'])
        point = [centroX, centroY]

        listaX = []
        listaY = []
        for tupla in cnt:
            tupla = tuple(tupla[0])
            listaX.append(tupla[0])
            listaY.append(tupla[1])
        minX = min(listaX)
        maxX = max(listaX)
        minY = min(listaY)
        maxY = max(listaY)

        A = ((minX+maxX)/2, minY)
        B = (minX, maxY)
        C = (maxX, maxY)
        corners = (A, B, C)

    else:
        print("SEM CONTORNO")
        centroX = 0
        centroY = 0
        cnt = 0
        point = [centroX, centroY]  
        A, B, C = 0, 0, 0  

    cv2.imshow('MaskYellow', mask)
    cv2.waitKey(1)   

    return A, B, C
