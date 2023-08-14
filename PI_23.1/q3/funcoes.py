#clicar no quadradinho com a lupinha para vizualizar o arquivo md

import cv2
import numpy as np

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

'''

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
xlist = contorno[0][:,:,0]
ylist = contorno[0][:,:,1]
x, y = (np.mean(xlist), np.mean(ylist))
P = (x,y) #centro

#BOUNDING RECT (recebe 1 cotorno e devolve a coordenada x e y do inicio e as dimensoes)
x, y, width, height = cv2.boundingRect(contornos[i])

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
mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((11, 11)))
mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((11, 11)))
'''

#MOBILE NET
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

CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
    "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
    "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
    "sofa", "train", "tvmonitor", "sports ball"]

CONFIDENCE = 0.7
#COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))

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
    
    
'''    
if len(resultados) > 0: 
    pi_dog = resultados[0][2]
    pf_dog = resultados[0][3]
    x = np.mean((pi_dog[0], pf_dog[0])) #n eh tupla
    y = np.mean((pi_dog[1], pf_dog[1])) #n eh tupla
    #cv2.circle(frame, (int(x), int(y)), 5, (255, 0, 0), -1) # Desenha o centro do cachorro 
'''
    
