#clicar no quadradinho com a lupinha para vizualizar o arquivo md

'''LINK HUE
file:///home/borg/Downloads/HUEcores.webp
'''

'''LINK COLORPICKER
https://igordsm.github.io/hsv-tool/
'''

import cv2
import numpy as np
from mobilenet import detect, net, CONFIDENCE, COLORS, CLASSES

#MASK BRANCA OU PRETA
def mask_white(img):
    gray = cv2.cvtColor(img.copy(), cv2.COLOR_BGR2GRAY)
    menor = 250
    maior = 255
    mask = cv2.inRange(gray, menor, maior)
    return mask

def mask_black(img):
    gray = cv2.cvtColor(img.copy(), cv2.COLOR_BGR2GRAY)
    menor = 0
    maior = 5
    mask = cv2.inRange(gray, menor, maior)
    return mask

#MASK COLORIDA
def mask_color(img, maior, menor):
    hsv = cv2.cvtColor(img.copy(), cv2.COLOR_BGR2HSV)
    upper = (maior//2, 255, 255)
    lower = (menor//2, 50, 50)
    mask = cv2.inRange(hsv, lower, upper)

    return mask 

#CONTORNOS (find, sorted, draw, area)
contornos, arvore = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
contornos_filtrados = sorted(contornos, key=cv2.contourArea)[-2:] #seleciona as duas maiores areas
contornos_desenhados = cv2.drawContours(img, contornos, -1, [255,0,0], 4)
area = cv2.contourArea(contornos[i]) 

#CENTROS
#desenhar centro
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

#encontrar e desenhar centros
def centros(img, contornos):
    """Não mude ou renomeie esta função
        deve receber uma lista de contornos e retornar, respectivamente,
        a imagem com uma cruz no centro de cada segmento e o centro de cada. 
        formato: img, x_list, y_list
    """

    img = img.copy()
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

#OUTRO JEITO DE ENCONTRAR
xlist = contornos[0][:,:,0]
ylist = contornos[0][:,:,1]
x, y = np.mean(xlist), np.mean(ylist)
P = (x,y) #centro

#BOUNDING RECT (recebe 1 cotorno e devolve a coordenada x e y do inicio e as dimensoes)
x, y, width, height = cv2.boundingRect(contornos[i])
w, h = img.shape

#DESENHOS
retangulo = cv2.rectangle(img, (x1, y1), (x2,y2), [255,0,0],4)
circulo = cv2.circle(frame, (int(x_centro), int(y_centro)), 5, (255, 255, 255), -1)
linha = cv2.line(img, (x1, x1), (x2, y2), (0,0,255), 5, cv2.LINE_AA)
txt_na_img = cv2.putText(img=img, text=texto, org=(10,20), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(0,0,255), thickness=2, lineType=cv2.LINE_AA)

#SOLUCAO PARA RODAR EM VIDEO
def solucao(bgr): #entra frame - img bgr
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    lower = np.array([0, 0, 200], nptype=np.uint8)
    upper = np.array([180, 37, 255])

#MORPHOLOGY EX
mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5)))
mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5)))

#COLOR SEGMENTATION
def color_segmentation(img, lower, upper):
    hsv = cv2.cvtColor(img.copy(), cv2.COLOR_BGR2HSV)

    lower_hsv = np.array([lower//2,50,50],dtype=np.uint8) #yellow
    upper_hsv = np.array([upper//2,255,255],dtype=np.uint8)
    kernel = np.ones((5,5),np.uint8)

    mask = cv2.inRange(hsv, lower_hsv, upper_hsv)
    mask = cv2.morphologyEx(mask,cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask,cv2.MORPH_CLOSE, kernel)

    # find contours
    contours,_ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    if len(contours) > 0:
        # maior contorno (maior area)
        cnt = max(contours, key = lambda x: cv2.contourArea(x))

        # centro
        M = cv2.moments(cnt)
        centroX = int(M['m10']/M['m00'])
        centroY = int(M['m01']/M['m00'])
        point = [centroX, centroY]
        

    else:
        centroX = 0
        centroY = 0
        cnt = 0
        point = [centroX, centroY]     

    #cv2.imshow('MaskYellow', mask)
    #cv2.waitKey(1)   

    return point, cnt

#MOBILE NET
_, resultadosMB = detect(net, img, CONFIDENCE, COLORS, CLASSES)
classeMB = resultadosMB[0][0]
xi, yi, xf, yf = resultadosMB[0][2][0], resultadosMB[0][2][1], resultadosMB[0][3][0], resultadosMB[0][3][1]
centroMB = ((xi+xf)/2, (yi+yf)/2)