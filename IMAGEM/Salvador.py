import cv2
#Escrever na img
def texto(img,a, p):
        """Escreve na img RGB dada a string a na posição definida pela tupla p"""
        cv2.putText(img, str(a), p, font,1,(255,255,255),2,cv2.LINE_AA)

#Exemplo
font = cv2.FONT_HERSHEY_PLAIN
texto(frame, "Japao",(x1,y1-10))


#Mask Branco
gray= cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
rgb= cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
cor_menor = 0
cor_maior = 240
mask= cv2.inRange(gray, cor_menor, cor_maior)
mask = cv2.bitwise_not(mask)
elemento_estruta = np.ones([10,10])
mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, elemento_estruta)


#Mask com qualquer cor (ColorPicker)
img_hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
hsv1, hsv2 = aux.ranges('#eeff00') #Alterar esse valor usando colorpicker
mask = cv2.inRange(img_hsv, hsv1, hsv2)
elemento_estrut = np.ones([10,10])
mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, elemento_estrut)

#Pegando coordenadas dos contornos
coordenadas = []
    for contorno in contornos:
        M = cv2.moments(contorno)
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
        coordenadas.append([cX,cY])


#Pegando os maiores contornos
contornos, arvore = cv2.findContours(mask_gray.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE) 
contornos = sorted(contornos, key=cv2.contourArea)[-4:] #4 maiores
#Exemplo
japao = contornos[2]
polonia = contornos[1]


#Achar retangulos/quadrados na imagem
x1, y1, w1, h1 = cv2.boundingRect('colocar o contorno aqui')
#x inicial, y inicial, w largura, h altura


#Encontrar circulos na imagem e o centro deles
gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, 200,param1=90,param2=30,minRadius=35,maxRadius=50)
    if circles is not None:
        circles = np.round(circles[0, :]).astype("int")
        for (x, y, r) in circles:
            #cv2.circle(frame, (x, y), r, (0, 255, 0), 4)
            cv2.rectangle(frame, (x - 5, y - 5), (x + 5, y + 5), (0, 128, 255), -1)
            x_bola = x
            y_bola = y


#Achar retas na imagem
bordas = aux.auto_canny(img)
lines = cv2.HoughLines(bordas, 1, np.pi/180, 100) 
#ou
lines = cv2.HoughLinesP(bordas, 10, math.pi/180.0, threshold=60, minLineLength=60, maxLineGap=10)
a,b,c = lines.shape
hough_img_rgb = cv2.cvtColor(bordas, cv2.COLOR_GRAY2BGR)
for i in range(a):
    # Faz uma linha ligando o ponto inicial ao ponto final, com a cor vermelha (BGR)
    cv2.line(hough_img_rgb, (lines[i][0][0], lines[i][0][1]), (lines[i][0][2], lines[i][0][3]), (0, 0, 255), 5, cv2.LINE_AA)


#Regressao Linear
from sklearn import linear_model
img = bgr.copy()    
reg = linear_model.LinearRegression()
ransac = linear_model.RANSACRegressor(reg)
yr = y_array[:-1].reshape(-1,1) # Entradas do modelo
xr = x_array[:-1].reshape(-1,) # saídas do modelo
try:
    ransac.fit(yr, xr)
    reg = ransac.estimator_
    coef_angular, coef_linear = reg.coef_, reg.intercept_
except:
    coef_angular = 0; coef_linear = 0

def f(y):
    x = coef_angular*y + coef_linear
    return x
def plot_regression_xfy(y, img, m, h, color=(255,0,0)):
    x = m*y + h
    y_min = int(min(y_array))
    y_max = 1000

    x_min = int(f(y_min))
    x_max = int(f(y_max)) 
    if x_min > 0 and y_min > 0:   
        cv2.line(img, (x_min, y_min), (x_max, y_max), color, thickness=3) 

plot_regression_xfy(y_array, img, coef_angular, coef_linear, color=(0,0,255))



#Mobile Net
CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
"bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
"dog", "horse", "motorbike", "person", "pottedplant", "sheep",
"sofa", "train", "tvmonitor"]
COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3)) # uma cor para cada class
def detect(net, frame, CONFIDENCE, COLORS, CLASSES):
    img = frame.copy()
    (h, w) = img.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(img, (300, 300)), 0.007843, (300, 300), 127.5)
    net.setInput(blob)
    detections = net.forward()
    resultados = [] 

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
            class_id = int(detections[0, 0, i, 1])
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            # display the prediction
            label = "{}: {:.2f}%".format(CLASSES[class_id], confidence * 100)
            print("[INFO] {}".format(label))
            cv2.rectangle(img, (startX, startY), (endX, endY),
                COLORS[class_id], 2)
            y = startY - 15 if startY - 15 > 15 else startY + 15
            cv2.putText(img, label, (startX, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[class_id], 2)

            resultados.append((CLASSES[class_id], confidence*100, (startX, startY),(endX, endY) ))

    return img, resultados

 # LOAD A MOBILE NET PARA IDENTIFICAR O objeto
net = cv2.dnn.readNetFromCaffe("MobileNetSSD_deploy.prototxt.txt", "MobileNetSSD_deploy.caffemodel") #Talvez mudar os nomes entre parenteses (ver no arquivo)

# Detectar o objeto
CONFIDENCE = 0.5
frame,resultados = detect(net, frame, CONFIDENCE, COLORS, CLASSES)
# Pega o CM do cachorro 
if len(resultados) > 0: 
    pi_dog = resultados[0][2]
    pf_dog = resultados[0][3]
    x = np.mean((pi_dog[0], pf_dog[0])) #n eh tupla
    y = np.mean((pi_dog[1], pf_dog[1])) #n eh tupla
    #cv2.circle(frame, (int(x), int(y)), 5, (255, 0, 0), -1) # Desenha o centro do cachorro 
    


