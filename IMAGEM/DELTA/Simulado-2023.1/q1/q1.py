#!/usr/bin/python3
# -*- coding: utf-8 -*-

# Este NÃO é um programa ROS

from __future__ import print_function, division 

import cv2
import os,sys, os.path
import numpy as np
from funcoes1 import color_segmentation, color_segmentation_white

print("Rodando Python versão ", sys.version)
print("OpenCV versão: ", cv2.__version__)
print("Diretório de trabalho: ", os.getcwd())

# Arquivos necessários
video = "bandeiras_movie.mp4"

######
corner_jp = ((10,10),(100,100))
corner_pl = ((5,5),(200,200))

# Responda dentro desta função. 
# Pode criar e chamar novas funções o quanto quiser
def encontra_japao_polonia_devolve_corners(bgr):
    frame = bgr.copy()

    #MASK BRANCA
    corners_jp, corners_pl = color_segmentation_white(frame) 
    
    if corners_jp != 0 and corners_pl != 0:
        #desenhando contornos
        cv2.rectangle(frame, (corners_jp[0], corners_jp[1]), (corners_jp[2], corners_jp[3]), [255,0,0],4)
        cv2.rectangle(frame, (corners_pl[0], corners_pl[1]), (corners_pl[2], corners_pl[3]), [255,0,0],4)
    else:
        print("SEM CONTORNO")

    return frame, corners_jp, corners_pl

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

        # Programe só na função encontra_japao_devolve_corners. Pode criar funções se quiser
        saida, japao, polonia = encontra_japao_polonia_devolve_corners(frame)

        print("Corners x-y Japao")
        print(japao)
        print("Corners x-y Polonia")
        print(polonia)

        # NOTE que em testes a OpenCV 4.0 requereu frames em BGR para o cv2.imshow
        cv2.imshow('imagem', saida)

        if cv2.waitKey(15) & 0xFF == ord('q'):
            break

    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()


