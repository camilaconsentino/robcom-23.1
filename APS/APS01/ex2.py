#!/usr/bin/python
# -*- coding: utf-8 -*-

# Este NÃO é um programa ROS

from __future__ import print_function, division 

import cv2
import os,sys, os.path
import numpy as np
import matplotlib.pyplot as plt

print("Rodando Python versão ", sys.version)
print("OpenCV versão: ", cv2.__version__)
print("Diretório de trabalho: ", os.getcwd())


def realca_caixa_vermelha(bgr): 
    res = bgr.copy()
    cor_rgb = cv2.cvtColor(res, cv2.COLOR_BGR2RGB)
    
    canal_v = cor_rgb[:,:,0]
    mask_red = np.zeros_like(canal_v)
    mask_red[canal_v > 10] = 255
    
    canal_g = cor_rgb[:,:,1]
    mask_green = np.zeros_like(canal_g)
    mask_green[canal_g > 50] = 255
    
    canal_b = cor_rgb[:,:,2]
    mask_blue = np.zeros_like(canal_b)
    mask_blue[canal_b < 20] = 255
    
    mask = cv2.bitwise_and(mask_blue, mask_red, mask=mask_green)
    plt.imshow(mask, cmap='gray')
    plt.show()
    
    return res


if __name__ == "__main__":
    img = cv2.imread("img/cena_canto_sala.jpg")
    
    # Faz o processamento
    saida = realca_caixa_vermelha(img)
    cv2.imwrite( "saida_ex2.png", saida)


    # NOTE que a OpenCV terminal trabalha com BGR
    cv2.imshow('entrada', img)

    cv2.imshow('saida', saida)

    cv2.waitKey()
    cv2.destroyAllWindows()

