import cv2
import funcoes
import numpy as np

def prateleira_arrumada(imagem_bgr):
    
    n_cima = 0
    n_baixo = 0
    cima_corretos = 0
    baixo_corretos = 0

    hsv = cv2.cvtColor(imagem_bgr, cv2.COLOR_BGR2HSV)

    amarelo = cv2.inRange(hsv, 25, 35)
    amerelo = cv2.morphologyEx(amarelo, cv2.MORPH_OPEN, np.ones((5, 5)))
    contornos_amarelos, _ = cv2.findContours(amarelo, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

    magenta = cv2.inRange(hsv, 290, 310)
    magenta = cv2.morphologyEx(magenta, cv2.MORPH_OPEN, np.ones((5, 5)))
    contornos_magentas, _ = cv2.findContours(magenta, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

    h_img = imagem_bgr.shape[0]

    for c in amarelo:
        x, y, w, h = cv2.boundingRect(c)
        if y < h_img//2:
            n_cima +=1
            if h > w:
                cima_corretos += 1   
        else:
            n_baixo += 1
    
    for c in magenta:
        x, y, w, h = cv2.boundingRect(c)
        if y < h_img//2:
            n_cima +=1   
        else:
            n_baixo += 1
            if w > h:
                baixo_corretos += 1
    
    return 0,0,0,0

if __name__ == '__main__':
    img = cv2.imread("img/teste1.png")
    v = prateleira_arrumada(img)

    print(f'''
    - {v[0]} produtos na prateleira de cima
    - {v[1]} estão arrumados (prateleira correta e orientação correta) na prateleira de cima
    - {v[2]} na prateleira de baixo
    - {v[3]} estão arrumados (prateleira correta e orientação correta) na prateleira de baixo
''')