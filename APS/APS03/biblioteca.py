#!/usr/bin/python
# -*- coding: utf-8 -*-

import cv2
import math
import numpy as np
from sklearn import linear_model

from sklearn.linear_model import LinearRegression

def segmenta_linha_amarela(bgr):
    """Não mude ou renomeie esta função
        deve receber uma imagem bgr e retornar uma máscara com os segmentos amarelos do centro da pista em branco.
        Utiliza a função cv2.morphologyEx() para limpar ruidos na imagem
    """

    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    mask_amarela = cv2.inRange(hsv, (45//2, 50, 50), (75//2, 255, 255))
    kernel = np.ones([3,3])
    opening = cv2.morphologyEx(mask_amarela, cv2.MORPH_OPEN, kernel)
    return opening

def encontrar_contornos(mask):
    """Não mude ou renomeie esta função
        deve receber uma imagem preta e branca e retornar todos os contornos encontrados
    """

    contornos, arvore = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
   
    return contornos

def crosshair(img, point, size, color):
    """ Desenha um crosshair centrado no point.
        point deve ser uma tupla (x,y)
        color é uma tupla R,G,B uint8
    """
    x,y = point
    x = int(x)
    y = int(y)
    cv2.line(img,(x - size,y),(x + size,y),color,2)
    cv2.line(img,(x,y - size),(x, y + size),color,2)

def encontrar_centro_dos_contornos(bgr, contornos):
    """Não mude ou renomeie esta função
        deve receber uma lista de contornos e retornar, respectivamente,
        a imagem com uma cruz no centro de cada segmento e o centro de cada. 
        formato: img, x_list, y_list
    """

    
    img = bgr.copy()
    x_list = []
    y_list = []

    for contorno in contornos:
        M = cv2.moments(contorno)
    
        x = (int(M["m10"]/M["m00"]))
        x_list.append(x)
        y = (int(M["m01"]/M["m00"]))
        y_list.append(y)

        cv2.drawContours(img, [contorno], -1, [255,0,0], 3)


        crosshair(img, (x,y), 3, [255,0,0])

    return img, x_list, y_list


def desenhar_linha_entre_pontos(bgr, X, Y, color):
    """Não mude ou renomeie esta função
        deve receber uma lista de coordenadas XY, e retornar uma imagem com uma linha entre os centros EM SEQUENCIA do mais proximo.
    """
    img = bgr.copy()

    for i in range(len(X)-1):
        cv2.line(img, (X[i], Y[i]), (X[i+1], Y[i+1]),[255,0,0], 2)

    return img

def regressao_por_centro(bgr, x_array, y_array):
    """Não mude ou renomeie esta função
        deve receber uma lista de coordenadas XY, e estimar a melhor reta, utilizando o metodo preferir, que passa pelos centros. Retorne a imagem com a reta e os parametros da reta
        
        Dica: cv2.line(img,ponto1,ponto2,color,2) desenha uma linha que passe entre os pontos, mesmo que ponto1 e ponto2 não pertençam a imagem.
    """
    img = bgr.copy()

    reg = linear_model.LinearRegression()
    yr = y_array.reshape(-1,1) # Entradas do modelo
    xr = x_array.reshape(-1,) # saídas do modelo
    reg.fit(yr,xr)

    coef_angular, coef_linear = reg.coef_, reg.intercept_

    def f(y):
        x = coef_angular*y + coef_linear
        return x
    
    def plot_regression_xfy(y, img, m, h, color=(255,0,0)):
        x = m*y + h
        y_min = int(min(y))-500 # precisa ser int para plotar na imagem
        y_max = int(max(y))+500

        x_min = int(f(y_min))
        x_max = int(f(y_max))    
        cv2.line(img, (x_min, y_min), (x_max, y_max), color=(0,0,255), thickness=3);       

    ransac = linear_model.RANSACRegressor(reg)
    ransac.fit(yr, xr)
    reg = ransac.estimator_
    coef_angular, coef_linear = reg.coef_, reg.intercept_

    plot_regression_xfy(y_array, img, coef_angular, coef_linear)

    return img, (coef_angular, coef_linear)

def calcular_angulo_com_vertical(img, lm):
    """Não mude ou renomeie esta função
        deve receber uma imagem contendo uma reta, 
        além do modelo de reggressão linear e determinar o ângulo da reta 
        com a vertical, em graus, utilizando o método que preferir.
    """
    coef_ang = lm[0]
    ang = math.atan(coef_ang)
    ang = math.degrees(ang)

    return ang

if __name__ == "__main__":
    print('Este script não deve ser usado diretamente')