#!/usr/bin/python
# -*- coding: utf-8 -*-

# Este NÃO é um programa ROS

from __future__ import print_function, division 

import cv2
import os,sys, os.path
import numpy as np

print("Rodando Python versão ", sys.version)
print("OpenCV versão: ", cv2.__version__)
print("Diretório de trabalho: ", os.getcwd())


def equaliza(gray): 
    res = gray.copy()
    
    xmin = np.min(res)
    xmax = np.max(res)
    a = 255/(xmax-xmin)
    b = -a*xmin
       
    for i in range(res.shape[0]):
        for j in range(res.shape[1]):
            res[i,j] = res[i, j] * a + b
     
    return res


if __name__ == "__main__":
    img = cv2.imread("img/RinTinTin.jpg")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Faz o processamento
    saida = equaliza(gray)
    cv2.imwrite("saida_ex1.png", saida)


    # NOTE que a OpenCV terminal trabalha com BGR
    cv2.imshow('entrada', img)

    cv2.imshow('saida', saida)

    cv2.waitKey()
    cv2.destroyAllWindows()
