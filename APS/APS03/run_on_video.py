#!/usr/bin/python
# -*- coding: utf-8 -*-

import cv2
import numpy as np
from biblioteca import *

print("Baixe o arquivo a seguir para funcionar: ")
print("https://raw.githubusercontent.com/Insper/robot22.1/main/aula03/yellow.mp4")

cap = cv2.VideoCapture('yellow.mp4')

while(True):
    # Capture frame-by-frame
    ret, img = cap.read()
    # frame = cv2.imread("frame0000.jpg")
    # ret = True
    
    if ret == False:
        print("Codigo de retorno FALSO - problema para capturar o frame")
        break
    else:

        mask = img.copy()
        
        segmentado = segmenta_linha_amarela(mask)
        contornos = encontrar_contornos(segmentado)

        if len(contornos) > 0:

            imagem, x, y = encontrar_centro_dos_contornos(mask, contornos)
            img_linha = desenhar_linha_entre_pontos(imagem, x, y, [255,0,0])
            if len (x) and len (y) > 1:
                X = np.array(x)
                Y = np.array(y)
                regressao_img, lm = regressao_por_centro(img_linha, X, Y)
                angulo = calcular_angulo_com_vertical(regressao_img, lm)

                texto = f"Angulo: {angulo:.2f} graus"
                cv2.putText(img=regressao_img, text=texto, org=(10,20), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(0,0,255), thickness=2, lineType=cv2.LINE_AA)

        else: 
            pass 

        img = regressao_img
        mask = segmentado
        
        # Imagem original
        cv2.imshow('img',img)
        # Mascara
        cv2.imshow('mask',mask)


        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()

