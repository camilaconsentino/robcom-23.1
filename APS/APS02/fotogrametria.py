#!/usr/bin/python
# -*- coding: utf-8 -*-

# Este NÃO é um programa ROS

import cv2
import os,sys, os.path
import numpy as np
import math

def encontrar_foco(D,H,h):
    f = h*D/H

    return f

def segmenta_circulo_ciano(hsv): 
    menor = (165//2, 50, 50)
    maior = (195//2, 255, 255)
    mask = cv2.inRange(hsv, menor, maior)
    
    return mask

def segmenta_circulo_magenta(hsv):
    menor = (285//2, 50, 50)
    maior = (315//2, 255, 255)
    mask = cv2.inRange(hsv, menor, maior)
    
    return mask

def encontrar_maior_contorno(segmentado):
    contornos, arvore = cv2.findContours(segmentado.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE) 
    if len(contornos) > 0:
        contorno = sorted(contornos, key=cv2.contourArea)[-1]
    else:
        return []
    
    return contorno

def encontrar_centro_contorno(contorno):
    M = cv2.moments(contorno)
    if M["m00"] > 0:
        Xcentro = int(M["m10"] / M["m00"])
        Ycentro = int(M["m01"] / M["m00"])
        
        return (Xcentro, Ycentro)
    else: 
        return (0,0)
def calcular_h(centro_ciano, centro_magenta):

    distancia = math.sqrt((centro_ciano[0] - centro_magenta[0])**2 + (centro_ciano[1] - centro_magenta[1])**2)
    return distancia

def encontrar_distancia(f,H,h):
    if h != 0:
        D = f/h*H
        return D
    else:
        return 0

def calcular_distancia_entre_circulos(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    segmentado_ciano = segmenta_circulo_ciano(hsv)
    contorno_ciano = encontrar_maior_contorno(segmentado_ciano)

    segmentado_magenta = segmenta_circulo_magenta(hsv)
    contorno_magenta = encontrar_maior_contorno(segmentado_magenta)

    if len(contorno_ciano) > 0:
        if len(contorno_magenta) > 0: 
            centro_magenta = encontrar_centro_contorno(contorno_magenta)
            centro_ciano = encontrar_centro_contorno(contorno_ciano)

            if centro_ciano>(0,0) and centro_magenta>(0,0):

                h = calcular_h(centro_ciano, centro_magenta)
            
                cv2.drawContours(img, [contorno_magenta], -1, [255, 0, 0], 3)
                cv2.drawContours(img, [contorno_ciano], -1, [255, 0, 0], 3)
                return h, centro_ciano, centro_magenta, img

    return 0,0,0,img


def calcular_angulo_com_horizontal_da_imagem(centro_ciano, centro_magenta):
    """Não mude ou renomeie esta função
        Deve calcular o angulo, em graus, entre o vetor formato com os centros do circulos e a horizontal.
    Entradas:
        centro_ciano - centro do círculo ciano no formato (X,Y)
        centro_magenta - centro do círculo magenta no formato (X,Y)
    Saídas:
        angulo - o ângulo entre os pontos em graus
    """
    if centro_ciano and centro_magenta:
        cxc = centro_ciano[0]
        cyc = centro_ciano[1]
        cxm = centro_magenta[0]
        cym = centro_magenta[1]
        
        vetor = ((cxc-cxm), (cyc-cym))
        moduloV = math.sqrt(vetor[0]**2+vetor[1]**2)
        '''calcular_h(centro_ciano, centro_magenta)'''
        
        horizontal = (1, 0)
        moduloH = 1
        
        produto_escalar = vetor[0]*horizontal[0] + vetor[1]*horizontal[1]
        angulo_rad = math.acos(produto_escalar/(moduloV*moduloH))
        
        angulo = angulo_rad * 180 / math.pi

    else:
        return 0
 
    return angulo
