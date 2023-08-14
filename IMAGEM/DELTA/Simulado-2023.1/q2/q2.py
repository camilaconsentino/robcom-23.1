#!/usr/bin/python
# -*- coding: utf-8 -*-

# Este NÃO é um programa ROS

from __future__ import print_function, division 

import cv2
import os,sys, os.path
import numpy as np
from funcoes2 import color_segmentation_star, color_segmentation_triangle
from triutil import point_in_triangle

print("Rodando Python versão ", sys.version)
print("OpenCV versão: ", cv2.__version__)
print("Diretório de trabalho: ", os.getcwd())

# Arquivos necessários
video = "triangulos.mp4"


if __name__ == "__main__":

    # Inicializa a aquisição da webcam
    cap = cv2.VideoCapture(video)

    print("Se a janela com a imagem não aparecer em primeiro plano dê Alt-Tab")

    while(True):
        # Capture frame-by-frame
        ret, frame = cap.read()
        
        if ret == False:
            #print("Codigo de retorno FALSO - problema para capturar o frame")
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue
            #sys.exit(0)

        # Our operations on the frame come here
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) 
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        #TODO: seu código vai aqui
        A, B, C = color_segmentation_triangle(frame, 105, 135)
        P = color_segmentation_star(frame, 225, 265)

        if A!=0 and B!=0 and C!=0 and P!=0:
            inTriangle = point_in_triangle(A, B, C, P)

            if inTriangle:
                cv2.putText(img=frame, text="DENTRO", org=(10,20), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(0,0,255), thickness=2, lineType=cv2.LINE_AA)
            else:
                cv2.putText(img=frame, text="FORA", org=(10,20), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(0,0,255), thickness=2, lineType=cv2.LINE_AA)
        else:
            print("SEM CONTORNO")

        # NOTE que em testes a OpenCV 4.0 requereu frames em BGR para o cv2.imshow
        cv2.imshow('imagem', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()


