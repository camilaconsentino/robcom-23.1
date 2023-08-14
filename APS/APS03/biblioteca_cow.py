#!/usr/bin/python
# -*- coding: utf-8 -*-

import cv2
import numpy as np
import math
import os
import argparse

# Check https://www.fypsolutions.com/opencv-python/ssdlite-mobilenet-object-detection-with-opencv-dnn/

COCO_labels = { 0: 'background',
    1: '"person"', 2: 'bicycle', 3: 'car', 4: 'motorcycle',
    5: 'airplane', 6: 'bus', 7: 'train', 8: 'truck', 9: 'boat',
    10: 'traffic light', 11: 'fire hydrant',12: 'street sign', 13: 'stop sign', 14: 'parking meter',
    15: 'zebra', 16: 'bird', 17: 'cat', 18: 'dog',19: 'horse',20: 'sheep',21: 'cow',22: 'elephant',
    23: 'bear', 24: 'zebra', 25: 'giraffe', 26: 'hat', 27: 'backpack', 28: 'umbrella',29: 'shoe',
    30: 'eye glasses', 31: 'handbag', 32: 'tie', 33: 'suitcase', 34: 'frisbee', 35: 'skis',
    36: 'snowboard', 37: 'sports ball', 38: 'kite', 39: 'baseball bat', 40: 'baseball glove',
    41: 'skateboard', 42: 'surfboard', 43: 'tennis racket', 44: 'bottle', 45: 'plate',
    46: 'wine glass', 47: 'cup', 48: 'fork', 49: 'knife', 50: 'spoon', 51: 'bowl', 52: 'banana',
    53: 'apple', 54: 'sandwich', 55: 'orange', 56: 'broccoli', 57: 'carrot', 58: 'hot dog', 59: 'pizza',
    60: 'donut', 61: 'cake', 62: 'chair', 63: 'couch', 64: 'potted plant', 65: 'bed', 66: 'mirror',
    67: 'dining table', 68: 'window', 69: 'desk', 70: 'toilet', 71: 'door', 72: 'tv', 73: 'laptop',
    74: 'mouse', 75: 'remote', 76: 'keyboard', 78: 'microwave', 79: 'oven', 80: 'toaster', 81: 'sink',
    82: 'refrigerator', 83: 'blender', 84: 'book', 85: 'clock', 86: 'vase', 87: 'scissors',
    88: 'teddy bear', 89: 'hair drier', 90: 'toothbrush', 91: 'hair brush'}

def load_mobilenet():
    """Não mude ou renomeie esta função
        Carrega o modelo e os parametros da MobileNet. 
        Retorna a rede carregada.
    """
    net = cv2.dnn.readNetFromCaffe("/home/borg/entregas-robotica/APS03/mobilenet_detection/MobileNetSSD_deploy.prototxt.txt", "/home/borg/entregas-robotica/APS03/mobilenet_detection/MobileNetSSD_deploy.caffemodel")
    return net


def detect(net, frame, CONFIDENCE, COLORS, CLASSES):
    """
        Recebe:
            net - a rede carregada
            frame - uma imagem colorida BGR
            CONFIDENCE - o grau de confiabilidade mínima da detecção
            COLORS - as cores atribídas a cada classe
            CLASSES - o array de classes
        Devolve: 
            img - a imagem com os objetos encontrados
            resultados - os resultados da detecção no formato
             [(label, score, point0, point1),...]
    """
    image = frame.copy()
    (h, w) = image.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 0.007843, (300, 300), 127.5)

    # pass the blob through the network and obtain the detections and
    # predictions
    print("[INFO] computing object detections...")
    net.setInput(blob)
    detections = net.forward()

    results = []

    # loop over the detections
    for i in np.arange(0, detections.shape[2]):
        # extract the confidence (i.e., probability) associated with the
        # prediction
        confidence = detections[0, 0, i, 2]

        # filter out weak detections by ensuring the `confidence` is
        # greater than the minimum confidence


        if confidence > CONFIDENCE:
            # extract the index of the class label from the `detections`,
            # then compute the (x, y)-coordinates of the bounding box for
            # the object
            idx = int(detections[0, 0, i, 1])
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            # display the prediction
            label = "{}: {:.2f}%".format(CLASSES[idx], confidence * 100)
            print("[INFO] {}".format(label))
            cv2.rectangle(image, (startX, startY), (endX, endY),
                COLORS[idx], 2)
            y = startY - 15 if startY - 15 > 15 else startY + 15
            cv2.putText(image, label, (startX, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[idx], 2)

            results.append((CLASSES[idx], confidence*100, (startX, startY),(endX, endY) ))

    # show the output image
    return image, results

def separar_caixa_entre_animais(img, resultados):
    """Não mude ou renomeie esta função
        recebe o resultados da MobileNet e retorna dicionario com duas chaves, 'vaca' e 'lobo'.
        Na chave 'vaca' tem uma lista de cada caixa que existe uma vaca, no formato: [ [min_X, min_Y, max_X, max_Y] , [min_X, min_Y, max_X, max_Y] , ...]. Desenhe um retângulo azul em volta de cada vaca
        Na chave 'lobo' tem uma lista de uma unica caixa que engloba todos os lobos da imagem, no formato: [min_X, min_Y, max_X, max_Y]. Desenhe um retângulo vermelho em volta dos lobos

    """

    img = img.copy()
    animais = {"vaca": [], "lobo": []}
    
    for item in resultados:
        if item[0] == "cow":
            animais['vaca'].append([item[2][0], item[2][1], item[3][0], item[3][1]])
            img = cv2.rectangle(img, (item[2][0], item[2][1]), (item[3][0], item[3][1]), (255,0,0),5) 
        elif item[0] == "horse" or item[0] == "sheep" or item[0] == "dog":
            animais['lobo'].append([item[2][0], item[2][1], item[3][0], item[3][1]])

    
    if len(animais['lobo']) > 1:
        listax = []
        listay = []
        for lobo in animais['lobo']:
            listax.append(lobo[0])
            listax.append(lobo[2])
            listay.append(lobo[1])
            listay.append(lobo[3])
        xmin = min(listax)
        ymin = min(listay)
        xmax = max(listax)
        ymax = max(listay)

        img = cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (0,0,255),5)   
        animais['lobo'] = [[xmin, ymin, xmax, ymax]] 
    else:
        xmin, ymin, xmax, ymax = animais['lobo'][0][0], animais['lobo'][0][1], animais['lobo'][0][2], animais['lobo'][0][3]
        img = cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (0,0,255),5)   
        animais['lobo'] = [[xmin, ymin, xmax, ymax]] 

    return img, animais

def calcula_iou(boxA, boxB):
    """Não mude ou renomeie esta função
        Calcula o valor do "Intersection over Union" para saber se as caixa se encontram
    """

    #CASO NAO HAJA NENHUM LOBO OU NENHUMA VACA
    if len(boxA)==0 or len(boxB)==0:
        return 0


    min_a = (boxA[0], boxA[1])
    max_a = (boxA[2], boxA[3])

    min_b = (boxB[0], boxB[1])
    max_b = (boxB[2], boxB[3])

    x1 = 0
    y1 = 0

    #VENDO QUAL CAIXA ESTA MAIS PARA CIMA/DIREITA:
    x1 = 0
    y1 = 0

    if min_a[0] > min_b[0]:
        x1 = min_a[0]
    else:
        x1 = min_b[0]
    
    if min_a[1] < min_b[1]:
        y1 = min_b[1]
    else:
        y1 = min_a[1]

    #VENDO QUAL CAIXA ESTA MAIS PARA BAIXO/ESQUERDA
    x2 = 0
    y2 = 0

    if max_a[0] < max_b[0]:
        x1 = max_a[0]
    else:
        x1 = max_b[0]
    
    if max_a[1] > max_b[1]:
        y1 = max_b[1]
    else:
        y1 = max_a[1] 


    P1 = (x1, y1)
    P2 = (x2, y2)

    overlapA = (x2-x1) * (y2-y1)
    areaA = (max_a[0]-min_a[0])*(max_a[1]-min_a[1])
    areaB = (max_b[0]-min_b[0])*(max_b[1]-min_b[1])
    unionA = areaB+areaA-overlapA

    iou = overlapA/unionA

    return iou

def checar_perigo(image, animais):
    """Não mude ou renomeie esta função
        Recebe as coordenadas das caixas, se a caixa de uma vaca tem intersecção com as do lobo, ela esta em perigo.
        Se estiver em perigo, deve escrever na imagem com a cor vermlha, se não, escreva com a cor azul.
        *Importante*: nesta função, não faça cópia da imagem de entrada!!
        
        Repita para cada vaca na imagem.
    """
    vaca = animais["vaca"][0]
    boxV = [vaca[0], vaca[1], vaca[2], vaca[3]]

    lobo = animais["lobo"][0]
    boxL = [lobo[0], lobo[1], lobo[2], lobo[3]]

    iou = calcula_iou(boxV, boxL)
    #cv2.putText(image, text=str(iou), org=(10,60), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(255,0,0), thickness=2, lineType=cv2.LINE_AA) 

    if iou < 0:
        cv2.putText(image, text="SEM PERIGO", org=(10,30), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(255,0,0), thickness=2, lineType=cv2.LINE_AA) 
    else:
        cv2.putText(image, text="PERIGO", org=(10,30), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(0,0,255), thickness=2, lineType=cv2.LINE_AA) 

    return image