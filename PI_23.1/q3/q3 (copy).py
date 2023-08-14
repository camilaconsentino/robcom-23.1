import cv2
import numpy as np
import math
import funcoes


def carrega():
    res = cv2.dnn.readNetFromCaffe("/home/borg/entregas-robotica/APS03/mobilenet_detection/MobileNetSSD_deploy.prototxt.txt", "/home/borg/entregas-robotica/APS03/mobilenet_detection/MobileNetSSD_deploy.caffemodel") 
    return res


def identificar_animais(res, imagem_bgr):
    animais = {
        'gatos': [],
        'cachorros': [],
        'passaros': []
    }

    CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
    "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
    "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
    "sofa", "train", "tvmonitor", "sports ball"]
    CONFIDENCE = 0.7
    COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))

    _, results = funcoes.detect(res, imagem_bgr, CONFIDENCE, COLORS, CLASSES)

    for item in results:
        if item[0] == "cat":
            animais['gatos'].append((item[2][0], item[2][1], item[3][0], item[3][1]))
            #img = cv2.rectangle(img, (item[2][0], item[2][1]), (item[3][0], item[3][1]), (255,0,0),5) 
        elif item[0] == "dog":
            animais['cachorros'].append((item[2][0], item[2][1], item[3][0], item[3][1]))
        elif item[0] == "bird":
            animais['passaros'].append((item[2][0], item[2][1], item[3][0], item[3][1]))

    return animais

def lista_perigos(animais):
    perigo = {
        'gatos': [],
        'cachorros': [],
        'passaros': []
    }


    #ACHANDO CENTRO DE CADA ANIMAL
    centro_gato = []
    centro_dog = []
    centro_bird = []
    for animal, posicoes in animais.items():
        if len(posicoes) > 0:
            if animal == 'gatos':
                for gato in posicoes:
                    x = (gato[0]+gato[2])//2
                    y = (gato[1]+gato[3])//2
                    centro_gato.append((x,y))
            elif animal == 'cachorros':
                for dog in posicoes:
                    x = (dog[0]+dog[2])//2
                    y = (dog[1]+dog[3])//2
                    centro_dog.append((x,y))
            elif animal == "passaros":
                for bird in posicoes:
                    x = (bird[0]+bird[2])//2
                    y = (bird[1]+bird[3])//2
                    centro_bird.append((x,y))

    #PERIGO GATOS:
    #gato perto de cachorro
    if len(centro_gato)>0 and len(centro_dog)>0:
        contador = 0
        for cat in centro_gato:
            for dog in centro_dog:
                distancia = math.sqrt((dog[0]-cat[0])**2 + (dog[1]-cat[1])**2)
                if distancia < 300:
                    perigo['gatos'].append(animais["gatos"][contador])
            contador+=1
    
    #PERIGO PASSAROS:
    #passaro perto de gato
    if len(centro_gato)>0 and len(centro_bird)>0:
        contador = 0 
        for cat in centro_gato:
            for bird in centro_bird:
                distancia = math.sqrt((bird[0]-cat[0])**2 + (bird[1]-cat[1])**2)
                if distancia < 300:
                    perigo['passaros'].append(animais["passaros"][contador])
            contador += 1


    return perigo

if __name__ == '__main__':
    img = cv2.imread("img/teste2.png")

    res = carrega()
    d = identificar_animais(res, img)
    print(d)

    print(lista_perigos(d))