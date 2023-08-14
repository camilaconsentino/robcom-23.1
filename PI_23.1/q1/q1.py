import cv2
import numpy as np
import funcoes

def identifica_bandeiras(imagem_bgr):
    '''
    Essa função deverá identificar as bandeiras na imagem passada como argumento
    e devolver uma lista de tuplas no formato

    ('pais', (x1, y1), (x2, y2))
    '''

    #TESTE1:
    lista = []

    #mascara branca
    mask_branca = funcoes.mask_white(imagem_bgr)

    #contornos
    contornos, arvore = cv2.findContours(mask_branca, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    contornos = sorted(contornos, key=cv2.contourArea)[-3:]
    #contornos = cv2.drawContours(imagem_bgr, contornos, -1, [255,0,0], 4)

    #cada contorno
    ct_peru = contornos[0]
    ct_monaco = contornos[2]
    ct_singapura = contornos[1]

    #corners PERU
    xP, yP, wP, hP = cv2.boundingRect(ct_peru)
    x1_p = xP-wP
    x2_p = xP+ 2*wP
    y1_p = yP
    y2_p = yP + hP
    cv2.rectangle(img, (x1_p, y1_p), (x2_p,y2_p), [255,0,0],4)
    lista.append(('Peru', (x1_p, y1_p), (x2_p,y2_p)))

    #corners MONACO
    xM, yM, wM, hM = cv2.boundingRect(ct_monaco)
    x1_m = xM
    x2_m = xM+wM
    y1_m = yM-hM
    y2_m = yM+hM
    cv2.rectangle(img, (x1_m, y1_m), (x2_m,y2_m), [255,0,0],4)
    lista.append(('Monaco', (x1_m, y1_m), (x2_m,y2_m)))

    #corners SINGAPURA
    xS, yS, wS, hS = cv2.boundingRect(ct_singapura)
    x1_s = xS
    x2_s = xS+wS
    y1_s = yS-hS
    y2_s = yS+hS
    cv2.rectangle(img, (x1_s, y1_s), (x2_s,y2_s), [255,0,0],4)
    lista.append(('Singapura', (x1_s, y1_s), (x2_s,y2_s)))

    return lista


if __name__ == '__main__':
    img = cv2.imread('img/teste1.png')
    items = identifica_bandeiras(img)

    for it in items:
        print(it)
        cv2.rectangle(img, it[1], it[2], (255, 0, 0), 5)
    
    cv2.imshow("Resultado", img)
    cv2.waitKey(0)
