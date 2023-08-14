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


def recorta_leopardo(bgr): 
    res = bgr.copy()
    cor_rgb = cv2.cvtColor(res, cv2.COLOR_BGR2RGB)
    canal_r = cor_rgb[:,:,0]
    canal_g = cor_rgb[:,:,1]
    canal_b = cor_rgb[:,:,2]
    
    #mascara vermelho
    mask_red = np.zeros_like(canal_r)
    mask_red[canal_r == 255] = 255
    
    mask_green = np.zeros_like(canal_g)
    mask_green[canal_g == 0] = 255
    
    mask_blue = np.zeros_like(canal_b)
    mask_blue[canal_b == 0] = 255
    
    mask = cv2.bitwise_and(mask_blue, mask_red, mask=mask_green)
    
    imagem = np.where(mask==255)
    linha = imagem[0]
    coluna = imagem[1]
    print(imagem)
    
    maxx = min(linha)
    maxy = min(coluna)
    
    #mascara azul  
    mask2_red = np.zeros_like(canal_r)
    mask2_red[canal_r == 0] = 255
    
    mask2_green = np.zeros_like(canal_g)
    mask2_green[canal_g == 0] = 255
    
    mask2_blue = np.zeros_like(canal_b)
    mask2_blue[canal_b == 255] = 255
    
    mask2 = cv2.bitwise_and(mask2_blue, mask2_red, mask=mask2_green)
    
    imagem2 = np.where(mask2==255)
    linha2 = imagem2[0]
    coluna2 = imagem2[1]
    print(imagem)
    
    minx = min(linha2)
    miny = min(coluna2)    
        
    #recorte    
    recorte = res[minx: maxx,miny: maxy]
    
    '''
    plt.imshow(mask, cmap='gray')
    plt.show()
    plt.imshow(mask2, cmap='gray')
    plt.show()
    '''
    
    return recorte


if __name__ == "__main__":
    img = cv2.imread("img/snowleopard_red_blue_600_400.png")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Faz o processamento
    saida = recorta_leopardo(img)
    cv2.imwrite("ex3_recorte_leopardo.png", saida)


    # NOTE que a OpenCV terminal trabalha com BGR
    cv2.imshow('entrada', img)

    cv2.imshow('saida', saida)

    cv2.waitKey()
    cv2.destroyAllWindows()

