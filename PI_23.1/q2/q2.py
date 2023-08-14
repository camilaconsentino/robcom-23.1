import cv2
import funcoes
import numpy as np

def prateleira_arrumada(imagem_bgr):
    '''
    Devolve 4 valores:

    - número de produtos na prateleira de cima
    - número de produtos que estão arrumados (prateleira correta e orientação correta) na prateleira de cima
    - número de produtos na prateleira de baixo
    - número de produtos que estão arrumados (prateleira correta e orientação correta) na prateleira de baixo
    '''
    
    p_cima = 0
    p_cima_corretos = 0
    p_baixo = 0
    p_baixo_corretos = 0

    #mask para as etiquetas 
    mask_yellow = funcoes.mask_color(imagem_bgr, 70, 50)
    #mask_yellow = cv2.morphologyEx(mask_yellow, cv2.MORPH_OPEN, np.ones((11, 11)))
    #mask_yellow = cv2.morphologyEx(mask_yellow, cv2.MORPH_CLOSE, np.ones((11, 11)))
    mask_magenta = funcoes.mask_color(imagem_bgr, 315, 285)
    
    #contornos
    ct_yellow, arvore = cv2.findContours(mask_yellow, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    #ct_yellow = sorted(ct_yellow, key=cv2.contourArea)[-4:]
    ct_magenta, arvore = cv2.findContours(mask_magenta, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    #ct_magenta = sorted(ct_magenta, key=cv2.contourArea)[-3:]

    #filtrando contornos
    area_yellow = cv2.contourArea(ct_yellow[0]) 
    cts_yellow = []
    for ct in ct_yellow:
        area = cv2.contourArea(ct)
        if abs(area - area_yellow) < 1:
            cts_yellow.append(ct)

    area_magenta = cv2.contourArea(ct_magenta[0]) 
    cts_magenta = []
    for ct in ct_magenta:
        area =  cv2.contourArea(ct)
        if abs(area - area_magenta) < 1:
            cts_magenta.append(ct)
    
    h_img = imagem_bgr.shape[0]
    
    #checando
    for ct in cts_yellow:
        x, y, w, h = cv2.boundingRect(ct)
        if y < h_img//2:
            p_cima +=1
            if h > w:
                p_cima_corretos += 1   
        else:
            p_baixo += 1
    
    for ct in cts_magenta:
        x, y, w, h = cv2.boundingRect(ct)
        if y < h_img//2:
            p_cima +=1   
        else:
            p_baixo += 1
            if w > h:
                p_baixo_corretos += 1
    
    #cv2.imshow("img", mask_yellow)
    #cv2.waitKey(0)
    
    return p_cima, p_cima_corretos, p_baixo, p_baixo_corretos

if __name__ == '__main__':
    img = cv2.imread("img/teste1.png")
    v = prateleira_arrumada(img)

    print(f'''
    - {v[0]} produtos na prateleira de cima
    - {v[1]} estão arrumados (prateleira correta e orientação correta) na prateleira de cima
    - {v[2]} na prateleira de baixo
    - {v[3]} estão arrumados (prateleira correta e orientação correta) na prateleira de baixo
''')