#!/usr/bin/python3
# -*- coding: utf-8 -*-

# Este NÃO é um programa ROS

from __future__ import print_function, division 

import cv2
import os,sys, os.path
import numpy as np
from funcao3 import color_segmentation
from mobilenet import detect, net, CONFIDENCE, COLORS, CLASSES
import math

print("Rodando Python versão ", sys.version)
print("OpenCV versão: ", cv2.__version__)
print("Diretório de trabalho: ", os.getcwd())

# Arquivos necessários
video = "dogtraining.mp4"

if __name__ == "__main__":

    # Inicializa a aquisição da webcam
    cap = cv2.VideoCapture(video)


    print("Se a janela com a imagem não aparecer em primeiro plano dê Alt-Tab")

    ultima_distancia = 100000000

    while(True):
        # Capture frame-by-frame
        ret, frame = cap.read()
        
        if ret == False:
            #print("Codigo de retorno FALSO - problema para capturar o frame")
            #cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            break
            

        # Our operations on the frame come here
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) 
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # TODO: seu código vai aqui

        centroBola = color_segmentation(frame, 49, 130)
        xB = centroBola[0]
        yB = centroBola[1]
        cv2.circle(frame, (int(xB), int(yB)), 10, (255, 255, 255), -1)


        #MOBILENET
        _, resultadosMB = detect(net, frame, CONFIDENCE, COLORS, CLASSES)
        if len(resultadosMB) > 0:
            classeMB = resultadosMB[0][0]
            xi, yi, xf, yf = resultadosMB[0][2][0], resultadosMB[0][2][1], resultadosMB[0][3][0], resultadosMB[0][3][1]
            centroMB = ((xi+xf)/2, (yi+yf)/2)
            xMB = centroMB[0]
            yMB = centroMB[1]

            distancia = math.sqrt((xB - xMB) ** 2 + (yB - yMB) ** 2)
            cv2.putText(frame, 
                        f'Distancia: {distancia:.2f}',
                        (0, 20), cv2.FONT_ITALIC,
                        1, (255, 255, 255), 4)
            
            if distancia > ultima_distancia:
                cv2.putText(frame, 
                        'CORRE',
                        (0, 50), cv2.FONT_ITALIC,
                        1, (255, 255, 255), 5)
            
            ultima_distancia = distancia 


        # NOTE que em testes a OpenCV 4.0 requereu frames em BGR para o cv2.imshow
        cv2.imshow('imagem', frame)

        # Pressione 'q' para interromper o video
        if cv2.waitKey(1000) & 0xFF == ord('q'):
            break

    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()

