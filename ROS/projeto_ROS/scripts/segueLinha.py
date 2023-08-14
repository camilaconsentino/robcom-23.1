#! /usr/bin/env python3
# -*- coding:utf-8 -*-

""" 
Running
    roslaunch my_simulation pista23-1.launch
    rqt_image_view
"""

import rospy
import numpy as np
import cv2
from geometry_msgs.msg import Twist, Vector3
from geometry_msgs.msg import Twist, PointStamped
from sensor_msgs.msg import LaserScan
from sensor_msgs.msg import CompressedImage
from cv_bridge import CvBridge,CvBridgeError
import numpy as np
from geometry_msgs.msg import Point
from sensor_msgs.msg import Image, CompressedImage
from std_msgs.msg import String
from nav_msgs.msg import Odometry
from tf.transformations import euler_from_quaternion
from std_msgs.msg import Float64
<<<<<<< HEAD
from mobilenet import detect, net, CONFIDENCE, COLORS, CLASSES
from math import asin
=======
from mobilenet import detect, net,CONFIDENCE, COLORS, CLASSES
import math 


>>>>>>> e92dfcdd25e6ce99af6328207f9df4b4ade9a20c

v = 0.1  # Velocidade linear
w = 0.5  # Velocidade angular


class Control():
    def __init__(self):
        
        self.rate = rospy.Rate(250) # 250 Hz
        
        self.robot_state = "aproxima"
        self.robot_machine = {
            #"procura": self.procura,
            "aproxima": self.aproxima,
            "gira1" : self.gira1,
            "aproxima2" : self.aproxima2,
            "segue" : self.segue,
            "aproximaVerde" : self.aproximaVerde,
            "centralizaCreeper" : self.centralizaCreeper, 
            "para" : self.para, 
            "pegaCreeper" : self.pegaCreeper,  
            "gira180" : self.gira180,
            "levaDropArea" : self.levaDropArea,
            "procuraDropArea" : self.procuraDropArea, 
            "dropCreeper" : self.dropCreeper,
            "voltaInicio" : self.voltaInicio,
            "aproximaDropArea" : self.aproximaDropArea 
        }

        # HSV Filter
        self.lower_hsv = np.array([14,200,233],dtype=np.uint8)
        self.upper_hsv = np.array([49,255,255],dtype=np.uint8)


        self.lower_hsvVerde = np.array([51, 50, 50],dtype=np.uint8)
        self.upper_hsvVerde = np.array([74, 255, 255],dtype=np.uint8)
        self.kernel = np.ones((5,5),np.uint8)

        # Image
        self.point = Point()
        self.point.x = -1
        self.point.y = -1
        self.start = None


        self.pointVerdex = None
        self.pointVerdey = None

        self.kp = 100
        self.w = 0
        self.cx = -1
        self.existencia = False
        self.contador = 0
        self.contador180 = 0
        self.contadorMB = 0
        self.bandeira = False
        self.bandeira2 = False
        self.bandeira3 = False
        self.bandeira4 = False
        self.entraAproxima2 = True  
        self.posInicial = 0

        # Subscribers
        self.bridge = CvBridge()
        self.image_sub = rospy.Subscriber('/camera/image/compressed',CompressedImage,self.image_callback,queue_size=1,buff_size = 2**24)
        self.laser_subscriber = rospy.Subscriber('/scan',LaserScan, self.laser_callback)

        # Publishers
        self.image_pub = rospy.Publisher("/image_publisher/", Image, queue_size=1)
        self.point_pub = rospy.Publisher("/center_publisher/", Point, queue_size=1)

        self.cmd_vel_pub = rospy.Publisher("cmd_vel", Twist, queue_size=3)
        self.cmd_vel_pub.publish(Twist())
        
        self.braco = rospy.Publisher("/joint1_position_controller/command", Float64, queue_size=1)
        self.pinca = rospy.Publisher("/joint2_position_controller/command", Float64, queue_size=1)

        #mobile net
        self.usaMB = False

        # RESETS DO PROGRAMA
        rospy.sleep(1)
        # Reseta a posição da garra 
        self.braco.publish(-1.0)
        self.pinca.publish(0.0)


        # # CARREGANDO A MOBILE NET
        # proto = "./mobilenet_detection/MobileNetSSD_deploy.prototxt.txt" # descreve a arquitetura da rede
        # model = "./mobilenet_detection/MobileNetSSD_deploy.caffemodel" # contém os pesos da rede em si

        # self.net = cv2.dnn.readNetFromCaffe(proto, model)

        # self.CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
        #     "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
        #     "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
        #     "sofa", "train", "tvmonitor"]

        # self.CONFIDENCE = 0.7
        # self.COLORS = np.random.uniform(0, 255, size=(len(self.CLASSES), 3))



    def odom_callback(self, data: Odometry):
        self.odom = data
        self.x = data.pose.pose.position.x
        self.y = data.pose.pose.position.y
        self.z = data.pose.pose.position.z
        
        # POSIÇÃO INICIAL
        if self.start == None:
            self.start = data.pose.pose.position

        

    def get_angular_error(self):
        x = self.point.x - self.x
        y = self.point.y - self.y
        theta = np.arctan2(y,x)

        self.distance = np.sqrt(x*2 + y*2)
        self.err = np.rad2deg(theta - self.yaw)
        self.err = (self.err + 100) % 360 - 100
        
        self.twist.angular.z = self.err * self.kp # Tirar daqui e colocar em outro lugar 
        
    def laser_callback(self, msg: LaserScan) -> None:
        self.laser_msg = np.array(msg.ranges).round(decimals=2) # Converte para np.array e arredonda para 2 casas decimais
        self.laser_msg[self.laser_msg == 0] = np.inf
        #self.frente = list(self.laser_msg)[0:5] + list(self.laser_msg)[-5:]
        print(f'Leitura na frente: {self.laser_msg[0]}')
        print("Faixa valida: ", msg.range_min , " - ", msg.range_max )

    def odom_callback(self, data: Odometry):
        #self.position = data.pose.pose.position
        
        orientation_list = [data.pose.pose.orientation.x,
                            data.pose.pose.orientation.y,
                            data.pose.pose.orientation.z,
                            data.pose.pose.orientation.w]

        self.odom = data
        self.x = data.pose.pose.position.x
        self.y = data.pose.pose.position.x

        if self.start == None:
            self.start = data.pose.pose.position

        self.roll, self.pitch, self.yaw = euler_from_quaternion(orientation_list)

        self.yaw = self.yaw % (2*np.pi)

    def image_callback(self, msg: CompressedImage) -> None:
        """
        Callback function for the image topic
        """
        try:
            cv_image = self.bridge.compressed_imgmsg_to_cv2(msg, "bgr8")
            img = cv_image.copy()
        except CvBridgeError as e:
            print(e)
        
        self.color_segmentation(cv_image) # Processamento da imagem

        if self.point.x != -1:
            self.cv_image = cv2.circle(cv_image, (self.point.x,self.point.y), 5, (0,0,255), -1)

        self.image_pub.publish(self.bridge.cv2_to_imgmsg(cv_image, "bgr8"))
        
        
        # MOBILE NET    
        _, self.resultados = detect(net, img, CONFIDENCE, COLORS, CLASSES)
        
        # Pega o tamanho da tela 
        self.w = cv_image.shape[1]
        self.cx = self.point.x


    

    def mobileNet_callback(self, msg: CompressedImage) -> None: 
        """
        Callback function for MOBILE NET
        """        

        try:
            cv_image = self.bridge.compressed_imgmsg_to_cv2(msg, "bgr8")
        except CvBridgeError as e:
            print(e)
        
        self.mb_image, self.resultados = detect(self, cv_image)

        self.image_pub.publish(self.bridge.cv2_to_imgmsg(self.mb_image, "bgr8"))

    # def detect(self, frame):
    #     """
    #         Recebe - uma imagem colorida BGR
    #         Devolve: objeto encontrado
    #     """
    #     image = frame.copy()
    #     (h, w) = image.shape[:2]
    #     blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 0.007843, (300, 300), 127.5)

    #     # pass the blob through the network and obtain the detections and
    #     # predictions
    #     print("[INFO] computing object detections...")
    #     self.net.setInput(blob)
    #     detections = self.net.forward()

    #     results = []

    #     # loop over the detections
    #     for i in np.arange(0, detections.shape[2]):
    #         # extract the confidence (i.e., probability) associated with the
    #         # prediction
    #         confidence = detections[0, 0, i, 2]

    #         # filter out weak detections by ensuring the `confidence` is
    #         # greater than the minimum confidence


    #         if confidence > self.CONFIDENCE:
    #             # extract the index of the class label from the `detections`,
    #             # then compute the (x, y)-coordinates of the bounding box for
    #             # the object
    #             idx = int(detections[0, 0, i, 1])
    #             box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
    #             (startX, startY, endX, endY) = box.astype("int")

    #             # display the prediction
    #             label = "{}: {:.2f}%".format(self.CLASSES[idx], confidence * 100)
    #             print("[INFO] {}".format(label))
    #             cv2.rectangle(image, (startX, startY), (endX, endY),
    #                 self.COLORS[idx], 2)
    #             y = startY - 15 if startY - 15 > 15 else startY + 15
    #             cv2.putText(image, label, (startX, y),
    #                 cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.COLORS[idx], 2)

    #             results.append((self.CLASSES[idx], confidence*100, (startX, startY),(endX, endY) ))

    #     # show the output image
    #     return image, results
    


    def color_segmentation(self,bgr: np.ndarray) -> None:
        #self.get_angular_error()
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, self.lower_hsv, self.upper_hsv)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self.kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self.kernel)
        mask2 = mask.copy()
        elemento_estrut = np.ones([10,10])
        mask2 = cv2.morphologyEx(mask2, cv2.MORPH_OPEN, elemento_estrut)
        self.contours,_ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        if len(self.contours) > 0:
            cnt = max(self.contours, key=lambda x: cv2.contourArea(x))
            M = cv2.moments(cnt)
            self.point.x = int(M['m10']/M['m00'])
            self.point.y = int(M['m01']/M['m00'])
        else:
            self.point.x = -1
        mask2 = mask2[:,int(float(mask.shape[0])/1.3):]
        # cv2.imshow("mask2",mask2)
        # cv2.imshow("mask",mask)
        self.contours2,_ = cv2.findContours(mask2, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        if len(self.contours2) > 0:
            cnt = max(self.contours2, key = lambda x: cv2.contourArea(x))
            self.existencia = True
        else:
            if self.existencia == True and len(self.contoursVerde)==0 and self.entraAproxima2 == True and self.bandeira3 == False and self.robot_state != "pegaCreeper" and self.robot_state != "gira180" and self.robot_state !="levaDropArea" and self.robot_state != "procuraDropArea" and self:
                self.robot_state = "aproxima2"
                print('\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n foi')
                self.existencia = False
                self.entraAproxima2 = False 

        maskVerde = cv2.inRange(hsv, self.lower_hsvVerde, self.upper_hsvVerde)
        maskVerde = cv2.morphologyEx(maskVerde, cv2.MORPH_OPEN, self.kernel)
        maskVerde = cv2.morphologyEx(maskVerde, cv2.MORPH_CLOSE, self.kernel)
        elemento_estrut = np.ones([10,10])
        maskVerde = cv2.morphologyEx(maskVerde, cv2.MORPH_OPEN, elemento_estrut)
        self.contoursVerde,_ = cv2.findContours(maskVerde, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        if len(self.contoursVerde) > 0:
            cnt = max(self.contoursVerde, key = lambda x: cv2.contourArea(x))
            M = cv2.moments(cnt)
            self.pointVerdex = int(M['m10']/M['m00'])
            self.pointVerdey = int(M['m01']/M['m00'])

        cv2.imshow("maskVerde",maskVerde)
        cv2.waitKey(1)     

    def aproxima(self) -> None:
        self.entraAproxima2 = True  
        self.twist.angular.z = 0
        self.twist.linear.x = 0.1
        self.entraAproxima2 = True 
        err = self.w/2 - self.point.x

    def aproxima2(self) -> None:
        self.contador +=0.1
        if self.contador >= 45 and self.bandeira == False and self.robot_state != "gira180":
            self.robot_state = "gira1"
        self.twist.angular.z = 0
        self.twist.linear.x = 0.2
        if len(self.contours) > 0:
            err = self.w/2 - self.point.x
            self.twist.angular.z = float(err)/self.kp
        errox = abs(abs(self.start.x) - abs(self.x))
        print(errox)
        if self.y > 0 and errox < 0.5 and self.bandeira == True and self.robot_state != "gira180":
            self.robot_state = "gira1"
        
        if (self.x - self.start.x < 0.5 and self.y - self.start.y < 0.5) and len(self.contours) == 0:
            self.robot_state = "para"
               
    def gira1(self) -> None:
        self.entraAproxima2 = True  
        self.twist = Twist()
        self.twist.angular.z = -0.2
        self.cmd_vel_pub.publish(self.twist)
        errox = abs(abs(self.start.x) - abs(self.x))
        if self.robot_state != "gira180" and self.robot_state != "levaDropArea" and self.robot_state != "procuraDropArea" and self.robot_state != "aproximaDropArea":
            if len(self.contours2) > 0 and self.y < 0 and self.robot_state != "aproximaVerde" and self.robot_state != "gira180" and self.robot_state != "levaDropArea"and self.robot_state != "procuraDropArea":
                self.robot_state = "segue"
            elif len(self.contours2) == 0 and self.y > 0 and errox < 0.5 and len(self.contoursVerde) == 0 and self.robot_state != "aproximaVerde" and self.robot_state != "gira180"and self.robot_state != "levaDropArea":
                self.robot_state = "segue"


    
    def segue(self) -> None:
        self.bandeira = True
        self.twist.angular.z = 0
        self.twist.linear.x = 0.1
        err = self.w/2 - self.point.x
        self.twist.angular.z = (float(err)/self.kp)
        if self.y > 0.5:
            self.bandeira3 == False
        else:
            self.bandeira3 == True
        if len(self.contoursVerde) > 0 and self.y > 0 and self.bandeira3 == False and self.robot_state != "pegaCreeper" and self.robot_state != "levaDropArea" and self.robot_state != "procuraDropArea"and self.robot_state != "gira180":
            self.robot_state = "aproximaVerde"
        
    def aproximaVerde(self) -> None:
        self.twist.angular.z = 0
        self.twist.linear.x = 0.1

        err = self.w/2 - self.pointVerdex
        self.twist.angular.z = (float(err)/self.kp)
        
        errox = (abs(self.pointVerdex) - abs(self.point.x))
        erroy = (abs(self.pointVerdey) - abs(self.point.y))

        print(self.laser_msg[0])

        if self.laser_msg[0] < 0.21 and self.robot_state == "aproximaVerde":
            self.twist.angular.z = 0
            self.twist.linear.x = 0
            print('foi')

            self.robot_state = "centralizaCreeper"

    def centralizaCreeper(self) -> None: 
        # Centraliza certinho no creeper antes de 
        err = (self.w/2)- self.pointVerdex
        self.twist.angular.z = (float(err)/self.kp) + 5
        #self.twist.linear.x = 0.05

        print(f'erro : {err}')
        if abs(err) < 8.0 and self.robot_state == "centralizaCreeper": # erro está em pixel 
            #self.twist.linear.x = 0.02
            self.twist.angular.z = 0.0
            self.twist.linear.x = 0.0
            self.robot_state = "pegaCreeper"
            
    def pegaCreeper(self) -> None:
        print('entrou pega creeper')
        # Robo volta a se aproximar
        #self.twist.linear.x = 0.05
        print(f'distancia: {self.laser_msg[0]}')
        #self.twist.linear.x  = 0.05

        
        self.twist.linear.x = 0.0
        rospy.sleep(0.5)

        #abre garra 
        self.pinca.publish(-1.0)
        rospy.sleep(0.5)

        #levanta braco para FRENTE
        self.braco.publish(-0.2)
        rospy.sleep(1)
    
        #fecha garra
        self.pinca.publish(0.0)
        rospy.sleep(0.5)

        #levanta braco
        self.braco.publish(1.5)
        rospy.sleep(0.5)

        self.robot_state = "gira180"
        
    def gira180(self) -> None:
        print('GIRA 180')
        self.twist = Twist()
        self.twist.angular.z = 0.2
        self.cmd_vel_pub.publish(self.twist)
        
        self.contador180 += 1
        print(f'CONTADOR 180: {self.contador180}')

        if self.contador180 > 3200:
            self.twist.angular.z = 0.0
            self.robot_state = 'levaDropArea'


    def levaDropArea(self) -> None:
        print('LEVANDO PARA DROP AREA')

        self.bandeira2 = True
        self.twist.angular.z = 0
        self.twist.linear.x = 0.2

        err = self.w/2 - self.point.x
        self.twist.angular.z = (float(err)/self.kp)

        # if self.y > 0.5:
        #     self.bandeira3 == False
        # else:
        #     self.bandeira3 == True

        # Contador para o robô la    
        self.contadorMB += 1 
        print(f'Contador MB: {self.contadorMB}') 

        if self.contadorMB > 3000 and self.robot_state != "pegaCreeper":
            # Para o robô de andar 
            self.twist.linear.x = 0.0
            self.robot_state = "procuraDropArea"
        

    def procuraDropArea(self) -> None:
        print( 'ENTROU DROP AREA')

        self.twist.angular.z = 0.2
        if len(self.resultados) != 0:
           if self.resultados[0][0] == "bicycle":
               self.twist.angular.z = 0.0 
               self.robot_state = "aproximaDropArea"
            
    def aproximaDropArea(self) -> None:
        print('ENTROU NA APROX DROP AREA')
        self.center_on_coord()

        if self.laser_msg[0] > 0.35:
            self.twist.linear.x = 0.1
        else:
            self.twist.linear.x = 0.0
            self.robot_state = "dropCreeper"


    def center_on_coord(self):
        print('CENTRALIZANDO VISÃO')
        # err começa setado em 0
        self.twist = Twist()
        err = 0

        print(self.resultados)
        # Identificar bike BOX MBN - só vai se o resultado da MBN for bike 
        if len(self.resultados) != 0:
            if self.resultados[0][0] == "bicycle":
                err = self.w/2 - ((self.resultados[0][2][0] + self.resultados[0][3][0])/2)

        # O quanto ele vai girar/cent em rel ao robo
        self.twist.angular.z = float(err) / self.kp


    def dropCreeper(self): 
        print('VAI DROPAR O CREEPER')


        #levanta braco para FRENTE
        self.braco.publish(-0.2)
        rospy.sleep(1)

        #abre garra 
        self.pinca.publish(-1.0)
        rospy.sleep(0.5)
        
        if self.laser_msg[0] < 0.45: #enquanto for maior que 0.5 vai pra frente e da velocidde linear de 0.2
            self.twist.linear.x = -0.2

        else: #quando for menor ou igual a 0.5 vai pra função parar
            self.twist.linear.x = 0.0

            #fecha garra
            self.pinca.publish(0.0)
            rospy.sleep(0.5)

            #abaixa o braco
            self.braco.publish(-1.0)
            rospy.sleep(0.5)

            self.robot_state = "voltaInicio"
    
    def voltaInicio(self): 
        print('VAI VOLTAR para POS Inicial ')

        #quando eu quero ir para uma coordenada especifica, tp a posicao inicial:
        anda = False 
        difx = self.start.x - self.x
        dify = self.start.y - self.y

        angulo = abs(math.asin(dify/((difx**2 + dify**2)**0.5)))
        if difx < 0 and dify < 0:
            angulo = np.pi + angulo
        elif difx < 0 and dify > 0:
            angulo = np.pi - angulo
        elif difx > 0 and dify < 0:
            angulo = 2*np.pi - angulo

        err = (angulo - self.yaw)

        if abs(err) > np.pi/6:
            self.twist.angular.z = 0.1

        else:
            anda = True
            self.twist.angular.z = float(err)*30/self.kp

        if abs(dify) <= 0.1 and abs(difx) <= 0.1:
            self.twist = Twist()
            self.robot_state = "para"


        elif anda:
            self.twist.linear.x = 0.1

    
    def para(self) -> None:
        self.twist.angular.z = 0
        self.twist.linear.x = 0

    def control(self) -> None:
        '''
        This function is called at least at {self.rate} Hz.
        This function controls the robot.
        Não modifique esta função.
        '''
        self.twist = Twist()
        print(f'self.robot_state: {self.robot_state}')	
        self.robot_machine[self.robot_state]()
        #print(f'self.robot_state: {self.robot_state}', self.twist)

        self.cmd_vel_pub.publish(self.twist)
        
        self.rate.sleep()


if __name__ == "__main__":
    #rospy.init_node("roda_exemplo")
    rospy.init_node("garra")  
    control = Control()
    try:
        while not rospy.is_shutdown():
            control.control()
    except rospy.ROSInterruptException:
        print("Ocorreu uma exceção com o rospy")