#!/usr/bin/python3
# -*- coding: utf-8 -*-

# Este NÃO é um programa ROS

from __future__ import print_function, division 

import cv2
import os,sys, os.path
import numpy as np
import math


print("Rodando Python versão ", sys.version)
print("OpenCV versão: ", cv2.__version__)
print("Diretório de trabalho: ", os.getcwd())

# Arquivos necessários
# Baixe o arquivo em:
# https://media.githubusercontent.com/media/Insper/robot20/master/media/dominoes.mp4
    
video = "dominoes.mp4"

#FUNCOES

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

        menor = 150
        maior = 255

        mask_branca = cv2.inRange(gray, menor, maior)

        contornos, arvore = cv2.findContours(mask_branca, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        contornos = sorted(contornos, key=cv2.contourArea)[-1:]

        
        x_lista = contornos[0][:,:,0]
        y_lista = contornos[0][:,:,1]

        x1, y1, x2, y2 = min(x_lista)[0], min(y_lista)[0], max(x_lista)[0], max(y_lista)[0]

        cv2.rectangle(frame, (x1,y1), (x2, y2), (0, 255, 0), 5)

        # inicio = (x1,y1)
        # fim = (x2, (y2//2))

        # inicio2 = (x1, (y2//2))
        # fim2 = (x2, y2)

        # cv2.rectangle(frame, inicio, fim, [0,255,0], 5)
        # cv2.rectangle(frame, inicio2, fim2, [0,255,255], 5)


        #pegando as bolinhas pretas 

        mask_preta = cv2.inRange(gray, 0, 50)

        contornos_pretos, arvore = cv2.findContours(mask_preta, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        contornos_pretos = sorted(contornos_pretos, key=cv2.contourArea)
        contornos_pretos = contornos_pretos[10:-2]

        contorno_ofc = []


        for contorno in contornos_pretos:
            if cv2.contourArea(contorno) > 600:
                contorno_ofc.append(contorno)
        

        cv2.drawContours(frame, contorno_ofc, -1, (0, 0, 255), 2)

        y_central = (max(y_lista) + min(y_lista))/2

        valorUp = 0
        valorDown = 0

        for c in contorno_ofc:
            M = cv2.moments(c)
            if M['m00'] != 0:
                x = int(M['m10']/M['m00'])
                y = int(M['m01']/M['m00'])

                if y < y_central - 10:
                    valorUp += 1
                elif y > y_central + 5:
                    valorDown +=1

        texto = f"{valorUp} por {valorDown}"
        cv2.putText(frame, text=texto, org=(10,20), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(0,0,255), thickness=2, lineType=cv2.LINE_AA) 





        # NOTE que em testes a OpenCV 4.0 requereu frames em BGR para o cv2.imshow
        cv2.imshow('imagem', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()


