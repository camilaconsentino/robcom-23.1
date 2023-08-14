#!/usr/bin/python
# -*- coding: utf-8 -*-

# Este NÃO é um programa ROS

import cv2
import os,sys, os.path
import numpy as np
import fotogrametria 

# ->>> !!!! FECHE A JANELA COM A TECLA ESC !!!! <<<<-

def calcular_angulo_e_distancia_na_image_da_webcam(img, f):
    """Não mude ou renomeie esta função
        ->>> !!!! FECHE A JANELA COM A TECLA ESC !!!! <<<<-
        deve receber a imagem da camera e retornar uma imagems com os contornos desenhados e os valores da distancia e o angulo.
    """
    h, centro_ciano, centro_magenta, img_contornos = fotogrametria.calcular_distancia_entre_circulos(img)

    if h is not 0:
        angulo = fotogrametria.calcular_angulo_com_horizontal_da_imagem(centro_ciano, centro_magenta)

        H = 12.70
        D = fotogrametria.encontrar_distancia(f,H,h)

        return img_contornos, D, angulo
    else:
        return img_contornos, 0, 0
    
def desenhar_na_image_da_webcam(img, distancia, angulo):
    """Não mude ou renomeie esta função
        ->>> !!!! FECHE A JANELA COM A TECLA ESC !!!! <<<<-
        deve receber a imagem da camera e retornar uma imagems com os contornos desenhados e a distancia e o angulo escrito em um canto da imagem.
    """
    texto = f"Distancia: {distancia:.2f}m Angulo: {angulo:.2f}graus"
    cv2.putText(img=img, text=texto, org=(10,20), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(0,0,255), thickness=2, lineType=cv2.LINE_AA) 

    return img

if __name__ == "__main__":
    cv2.namedWindow("preview")
    vc = cv2.VideoCapture(0)

    ## -> Mude o Foco <- ##
    f = fotogrametria.encontrar_foco(1, 1, 1)

    if vc.isOpened(): # try to get the first frame
        rval, frame = vc.read()
    else:
        rval = False

    while rval:
        img, distancia, angulo = calcular_angulo_e_distancia_na_image_da_webcam(frame, f)
        img = desenhar_na_image_da_webcam(img, distancia, angulo)
        cv2.imshow("preview", img)
        rval, frame = vc.read()
        key = cv2.waitKey(20)
        if key == 27: # exit on ESC
            break

    cv2.destroyWindow("preview")
    vc.release()

