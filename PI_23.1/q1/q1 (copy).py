import cv2
import numpy as np
import funcoes

def identifica_bandeiras(imagem_bgr):
    gray = cv2.cvtColor(imagem_bgr, cv2.COLOR_BGR2GRAY)
    bandeiras = np.zeros_like(gray)
    bandeiras[gray > 10] = 255 

    contornos_bandeiras = cv2.findContours(bandeiras, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

    hsv = cv2.cvtColor(imagem_bgr, cv2.COLOR_BGR2HSV)

    l = []
    for c in contornos_bandeiras:
        m_uma_bandeira = np.zeros_like(gray)
        cv2.drawContours(m_uma_bandeira, [c], -1, [255,0,0], 4)

        rect = cv2.boundingRect(c)

        bandeira = ''

        uma_bandeira_hsv = cv2.bitwise_and(hsv, hsv, mask=m_uma_bandeira)

        #IDENTIFICANDO A BANDEIRA DA IRLANDA
        laranja = cv2.inRange(uma_bandeira_hsv, lower, upper)
        n_laranja = np.sum(laranja) / 255
        if n_laranja > 0:
            bandeira = 'irlanda'

        l.append((bandeira, (rect[0], rect[1]), (rect[2]+rect[], rect[3])))

    return l


if __name__ == '__main__':
    img = cv2.imread('img/teste1.png')
    items = identifica_bandeiras(img)

    for it in items:
        print(it)
        cv2.rectangle(img, it[1], it[2], (255, 0, 0), 5)
    
    cv2.imshow("Resultado", img)
    cv2.waitKey(0)
