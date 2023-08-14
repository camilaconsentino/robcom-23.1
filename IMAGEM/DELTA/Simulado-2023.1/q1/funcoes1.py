#clicar no quadradinho com a lupinha para vizualizar o arquivo md

import cv2
import numpy as np 

#COLOR SEGMENTATION
def color_segmentation(img, lower, upper):
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

        contorno = contours[1] #maior contorno
        listaX = []
        listaY = []
        x, y, width, height = cv2.boundingRect(cnt)
        for tupla in contorno:
            tupla = tuple(tupla[0])
            listaX.append(tupla[0])
            listaY.append(tupla[1])
        minX = min(listaX)
        maxX = max(listaX)
        minY = min(listaY) - height
        maxY = max(listaY)
        corners = (minX, minY, maxX, maxY)

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
        corners = 0         

    cv2.imshow('MaskYellow', mask)
    cv2.waitKey(1)   

    #centro e area do maior contorno
    return point, corners

#COLOR SEGMENTATION WHITE
def color_segmentation_white(img):
    gray = cv2.cvtColor(img.copy(), cv2.COLOR_BGR2GRAY)
    menor = 250
    maior = 255
    mask = cv2.inRange(gray, menor, maior)

    kernel = np.ones((5,5),np.uint8)
    mask = cv2.morphologyEx(mask,cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask,cv2.MORPH_CLOSE, kernel)

    # find contours
    contours,_ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    if len(contours) > 0:
        # maior contorno (maior area)
        cnts = sorted(contours, key=cv2.contourArea)[-2:]

        #POLONIA
        listaX = cnts[0][:,:,0]
        listaY = cnts[0][:,:,1]
        minX = int(min(listaX))
        maxX = int(max(listaX))
        minY = int(min(listaY)) 
        maxY = int(2*max(listaY)-minY)
        cornersPL = (minX, minY, maxX, maxY)

        #JAPAO
        listaX = cnts[1][:,:,0]
        listaY = cnts[1][:,:,1]
        minX = int(min(listaX))
        maxX = int(max(listaX))
        minY = int(min(listaY)) 
        maxY = int(max(listaY))
        cornersJP = (minX, minY, maxX, maxY)
        

    else:
        cornersJP, cornersPL = 0, 0    

    cv2.imshow('MaskYellow', mask)
    cv2.waitKey(1)   

    #centro e area do maior contorno
    return cornersJP, cornersPL
