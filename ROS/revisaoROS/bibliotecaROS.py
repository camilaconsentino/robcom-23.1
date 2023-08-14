''' RODAR GARRA
roslaunch mybot_description mybot_control2.launch
'''

'''LINK HUE
file:///home/borg/Downloads/HUEcores.webp
'''

'''LINK COLORPICKER
https://igordsm.github.io/hsv-tool/
'''

#IMPORTS
import numpy as np
from numpy import random
from math import asin
import cv2
import rospy
from cv_bridge import CvBridge, CvBridgeError
from tf.transformations import euler_from_quaternion
from sensor_msgs.msg import LaserScan,CompressedImage
from geometry_msgs.msg import Twist
from geometry_msgs.msg import Point
from nav_msgs.msg import Odometry
from std_msgs.msg import Float64
from mobilenet import detect, net, CONFIDENCE, COLORS, CLASSES
import cv2.aruco as aruco


class Control():
    def __init__(self):
        self.rate = rospy.Rate(250) # 250 Hz

        #SUBSCRIBERS
        self.bridge = CvBridge()
        self.laser_subscriber = rospy.Subscriber('/scan',LaserScan, self.laser_callback)
        self.image_sub = rospy.Subscriber('/camera/image/compressed', CompressedImage, self.image_callback, queue_size=1, buff_size = 2**24)
        self.odom_sub = rospy.Subscriber('/odom', Odometry, self.odom_callback, queue_size=1)

        #PUBLISHERS
        self.cmd_vel_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=3)
        self.cmd_vel_pub.publish(Twist()) #se eu to fazendo isso aqui, nao preciso fzr no resto do codigo??

        #para a garra
        self.braco = rospy.Publisher("/joint1_position_controller/command", Float64, queue_size=1)
        self.garra = rospy.Publisher("/joint2_position_controller/command", Float64, queue_size=1)	

        #ESTADOS
        self.state = 1
        self.selected_mod = None
        self.robot_state = "anda"
        self.robot_machine = {
            "rotate_direita": self.rotate_direita,
            "rotate_esquerda": self.rotate_esquerda,
            "checar": self.checar,
            "center_on_coord": self.center_on_coord,
            "para": self.para,
            "aproxima_pista": self.aproxima_pista,
            "anda" : self.anda,
            "go_to_coord" : self.go_to_coord,
            "aproxima_aruco" : self.aproxima_aruco,
            "aproxima_caixa" : self.aproxima_caixa,
            
        }


        #VARIAVEIS
        self.twist = Twist()
        self.kp = 100 #(ou 200, qnt maior, mais preciso)
        self.kernel = np.ones((5,5),np.uint8) #para fazer morphologyEx
        self.initial_position = 0

        #para as masks
        self.color_param = {
            "magenta": {
                "lower": 315,
                "upper": 285
            },
            "yellow": {
                "lower": 45,                
                "upper": 75
            },
            "green": {
                "lower": 105,
                "upper": 135
            },	
            "blue": {
                "lower": 225,
                "upper": 265
            }
        }

        #ARUCO:
        #camera 
        self.camera_distortion = np.loadtxt('/home/borg/catkin_ws/src/meu_projeto/scripts/aruco/cameraDistortion_realsense.txt', delimiter=',')
        self.camera_matrix = np.loadtxt('/home/borg/catkin_ws/src/meu_projeto/scripts/aruco/cameraMatrix_realsense.txt', delimiter=',')

    #CALLBACKS
    def image_callback(self, msg: CompressedImage) -> None:
        """
        Callback function for the image topic
        """
        try:
            cv_image = self.bridge.compressed_imgmsg_to_cv2(msg, "bgr8")
            hsv = cv2.cvtColor(cv_image, cv2.COLOR_BGR2HSV)
            img = cv_image.copy()
        except CvBridgeError as e:
            print(e)

        #dimensao
        h, w, d = img.shape
        self.centro_segue = (h, 25*h//40) # corte que eu fiz só para que ele olhe para a pista
        self.centro_img = (w//2, h//2) # corte que fiz pra ele olhar o centro da imagem integralmente

        # color segmentation
        self.centro_yellow, mask_yellow, area_yellow = self.color_segmentation(hsv, self.color_param["yellow"]["lower"], self.color_param["yellow"]["upper"])
        
        #mobile net
        _,self.resultadosMB = detect(net, img, CONFIDENCE, COLORS, CLASSES)
        self.classeMB = self.resultadosMB[0][0]
        self.centroMB = (self.resultadosMB[0][2][0] + self.resultadosMB[0][3][0])/2 # CENTRO X

        #aruco
        self.ids, self.centros, self.distancia_aruco = self.geraAruco(cv_image)

    def laser_callback(self, msg: LaserScan) -> None:
        self.laser_msg = np.array(msg.ranges).round(decimals=2)
        self.laser_msg[self.laser_msg == 0] = np.inf

        self.laser_forward = np.min(list(self.laser_msg[0:5]) + list(self.laser_msg[354:359])) 
        self.laser_backwards = np.min(list(self.laser_msg[175:185]))

    def odom_callback(self, data: Odometry):
        self.position = data.pose.pose.position

        if self.initial_position == 0:
            self.initial_position = self.position
        
        orientation_list = [data.pose.pose.orientation.x,
                            data.pose.pose.orientation.y,
                            data.pose.pose.orientation.z,
                            data.pose.pose.orientation.w]

        self.roll, self.pitch, self.yaw = euler_from_quaternion(orientation_list)

        self.yaw = self.yaw % (2*np.pi)

    #FUNCOES
    def color_segmentation(self, hsv: np.ndarray, lower, upper) -> Point:

        self.lower_hsv = np.array([lower//2,50,50],dtype=np.uint8) #yellow
        self.upper_hsv = np.array([upper//2,255,255],dtype=np.uint8)

        mask = cv2.inRange(hsv, self.lower_hsv, self.upper_hsv)
        mask = cv2.morphologyEx(mask,cv2.MORPH_OPEN, self.kernel)
        mask = cv2.morphologyEx(mask,cv2.MORPH_CLOSE, self.kernel)

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
            existencia = True
            

        else:
            centroX = 0
            centroY = 0
            point = [centroX, centroY]     
            existencia = False 

        #cv2.imshow('MaskYellow', mask)
		#cv2.waitKey(1)   

        return point, existencia
    
    def geraAruco(self,img):
        centros = []
        distancia_aruco=[]
        # Gera mask Cinza
        grayColor = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        #Gera Dicionario com Arucos
        dicionarioAruco = aruco.getPredefinedDictionary(aruco.DICT_6X6_250)
        #Detecta Arucos e Gera variaveis
    
        cornersList, ids, _ = aruco.detectMarkers(
            grayColor, dicionarioAruco)
        if ids is not None:
            for i in range(len(ids)):
                if ids[i]<99:
                    ret = aruco.estimatePoseSingleMarkers(cornersList[i], 19, self.camera_matrix, self.camera_distortion)
                    rvec, tvec = ret[0][0,0,:], ret[1][0,0,:]
                    distancia_aruco.append(np.linalg.norm(tvec))
                else: 
                    ret = aruco.estimatePoseSingleMarkers(cornersList[i], 6, self.camera_matrix, self.camera_distortion)
                    rvec, tvec = ret[0][0,0,:], ret[1][0,0,:]
                    distancia_aruco.append(np.linalg.norm(tvec))
                

            for corners in cornersList:
                for corner in corners:
                    centros.append(np.mean(corner, axis=0))

        return ids, centros, distancia_aruco
            
    def center_on_coord(self) -> None:        

        if "atvd" == "seguir_pista":
            err = self.centro_segue[0] - self.centro_pista #centro do contorno/ da mask da pista
        elif "atvd" == "aproximar":
            err = self.centro_img[0] - self.centro_objeto #centro do aruco, do cnt do bloco, do mobile net...

        self.twist.angular.z = float(err)/self.kp

    def aproxima(self) -> None:
        self.center_on_coord #ajuste da vel angular a partir do erro, vai alinhando

        if self.laser_forward >= 0.3:
            self.twist.linear.x = 0.2
        else:
            self.twist.linear.x = 0.0
            #ou self.robot_state = "para"

    def go_to_coord(self) -> None:
        #quando eu quero ir para uma coordenada especifica, tp a posicao inicial:

        anda = False
        difx = self.initial_position.x - self.position.x
        dify = self.initial_position.y - self.position.y

        angulo = abs(asin(dify/((difx**2 + dify**2)**0.5)))
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

        elif anda:
            self.twist.linear.x = 0.1

    def get_angular_error(self, point): # funcao geral de get angular error para não ficar repetindo necessario mandar o point que ele precisa ir
        x = point.x - self.x
        y = point.y - self.y
        theta = np.arctan2(y , x)

        self.distance = np.sqrt(x*2 + y*2)
        self.err = np.rad2deg(theta - self.yaw)
        self.err = (self.err + 180) % 360 - 180

        self.twist.angular.z = self.err * self.kp

    def volta_inicio(self) -> None:
        self.get_angular_error(self.initial_position)

        if self.distance > 0.1:
            self.twist.linear.x = np.min([self.distance, 0.2])
        else:
            self.robot_state = "para"
    #bota um else para parar


    def garra(self) -> None:
        self.braco.publish(1.5) ## para baixo
        rospy.sleep(0.5)
        self.braco.publish(-1.5) ## para cima
        rospy.sleep(0.5)
        self.braco.publish(0.0) ## para frente
        rospy.sleep(0.5)
        self.garra.publish(-1.0) #aberta?
        rospy.sleep(0.5)
        self.garra.publish(0.0) #fechada
        rospy.sleep(0.5)

    def control(self):
        '''
        This function is called at least at {self.rate} Hz.
        This function controls the robot.
        '''
        self.twist = Twist()
        print(f'self.robot_state: {self.robot_state}')
        self.robot_machine[self.robot_state]()

        self.cmd_vel_pub.publish(self.twist)
        
        self.rate.sleep()

def main():
	rospy.init_node('Aleatorio')
	control = Control()
	rospy.sleep(1)

	while not rospy.is_shutdown():
		control.control()

if __name__=="__main__":
	main()