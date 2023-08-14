#!/usr/bin/python
# -*- coding: utf-8 -*-

import cv2
import numpy as np
import math
import matplotlib.pyplot as plt
import auxiliar as aux

def segmenta_linha_branca(bgr):
    """Não mude ou renomeie esta função
        deve receber uma imagem e segmentar as faixas brancas
    """
    #converter par gray, porque o branco seria o valor maximo.
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    menor = 250
    maior = 255

    mask_branca = cv2.inRange(gray, menor, maior)

    #plt.imshow(mask_branca)

    #res = np.zeros(bgr.shape[:-1], dtype=np.uint8)

    return mask_branca

def estimar_linha_nas_faixas(img, mask):
    """Não mude ou renomeie esta função
        deve receber uma imagem preta e branca e retorna dois pontos que formen APENAS uma linha em cada faixa. Desenhe cada uma dessas linhas na iamgem.
         formato: [[(x1,y1),(x2,y2)], [(x1,y1),(x2,y2)]]
    """
    bordas = aux.auto_canny(mask)
    lines = cv2.HoughLinesP(bordas, 1, math.pi/180.0, threshold=60, minLineLength=60, maxLineGap=10)

    hough_img_rgb = cv2.cvtColor(bordas, cv2.COLOR_GRAY2BGR)

    for i in (2, 7):
        # Faz uma linha ligando o ponto inicial ao ponto final, com a cor vermelha (BGR)
        cv2.line(img, (lines[i][0][0], lines[i][0][1]), (lines[i][0][2], lines[i][0][3]), (0, 0, 255), 5, cv2.LINE_AA)

    return [[(lines[2][0][0], lines[2][0][1]), (lines[2][0][2], lines[2][0][3])], [(lines[7][0][0], lines[7][0][1]), (lines[7][0][2], lines[7][0][3])]]

def calcular_equacao_das_retas(linhas):
    """Não mude ou renomeie esta função
        deve receber dois pontos que estejam em cada uma das
        faixas e retornar as equacões das duas retas, 
        onde y = h + m * x. Formato: [(m1,h1), (m2,h2)]
    """
    #pontos reta a
    x1a = linhas[0][0][0]
    y1a = linhas[0][0][1]
    x2a = linhas[0][1][0]
    y2a = linhas[0][1][1]

    #pontos reta b
    x1b = linhas[1][0][0]
    y1b = linhas[1][0][1]
    x2b = linhas[1][1][0]
    y2b = linhas[1][1][1]

    #equacao reta a
    ma = (y2a-y1a)/(x2a-x1a)
    ba = y1a - ma*x1a

    #equacao reta b
    mb = (y2b-y1b)/(x2b-x1b)
    bb = y1b - mb*x1b

    return [(ma,ba),(mb,bb)]

def calcular_ponto_de_fuga(img, equacoes):
    """Não mude ou renomeie esta função
        deve receber duas equacoes de retas e retornar o ponto de encontro entre elas. Desenhe esse ponto na imagem.
    """
    m1 = equacoes[0][0]
    b1 = equacoes[0][1]
    m2 = equacoes[1][0]
    b2 = equacoes[1][1]

    x = int((b2-b1)/(m1-m2))
    y = int(m1*x+b1)

    cv2.circle(img, (x,y), radius=5, color=(0, 0, 255), thickness=-1)

    return img, (x,y)

        
