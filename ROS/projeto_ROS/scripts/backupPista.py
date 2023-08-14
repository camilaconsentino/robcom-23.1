#! /usr/bin/env python3
# -*- coding:utf-8 -*-

""" 
Running
    roslaunch my_simulation pista23-1.launch
    rosrun aps4 pista.py
    rqt_image_view
    rostopic pub -1 /reset std_msgs/Empty "{}"
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
            "aproximaVerde" : self.aproximaVerde,
            "gira2" : self.gira2,
            "para" : self.para,
            "gira3" : self.gira3,
            "derrube" : self.derrube,
            "aproxima3": self.aproxima3
        }

        # HSV Filter
        #self.lower_hsv = np.array([15, 90, 90],dtype=np.uint8)
        #self.upper_hsv = np.array([35, 255, 255],dtype=np.uint8)

        # self.lower_hsv2 = np.array([25, 55, 174],dtype=np.uint8)
        # self.upper_hsv2 = np.array([73, 137, 255],dtype=np.uint8)

        # self.lower_hsv3 = np.array([28, 10, 226],dtype=np.uint8)
        # self.upper_hsv3 = np.array([48, 106, 255],dtype=np.uint8)

        # self.lower_hsv4 = np.array([28, 28, 151],dtype=np.uint8)
        # self.upper_hsv4 = np.array([48, 104, 255],dtype=np.uint8)

        self.lower_hsv = np.array([15, 90, 90],dtype=np.uint8)
        self.upper_hsv = np.array([35, 255, 255],dtype=np.uint8)

        # self.lower_hsv2 = np.array([0, 30, 196],dtype=np.uint8)
        # self.upper_hsv2 = np.array([50, 153, 255],dtype=np.uint8)

        # self.lower_hsv3 = np.array([29, 30, 207],dtype=np.uint8)
        # self.upper_hsv3 = np.array([54, 146, 255],dtype=np.uint8)

        # self.lower_hsv4 = np.array([8, 113, 188],dtype=np.uint8)
        # self.upper_hsv4 = np.array([44, 159, 255],dtype=np.uint8)



        self.lower_hsvVerde = np.array([13, 80, 29],dtype=np.uint8)
        self.upper_hsvVerde = np.array([67, 255, 110],dtype=np.uint8)
        self.kernel = np.ones((5,5),np.uint8)

        # Image
        self.point = Point()
        self.point.x = -1
        self.point.y = -1
        self.start = None
        self.x = 0
        self.y = 0


        self.pointVerdex = None
        self.pointVerdey = None

        self.startVerdex = None

        self.kp = 1000
        self.w = 0
        self.cx = -1
        self.existencia = False
        self.contador = 0
        self.contador2 = 0
        self.contador3 = 0
        self.contador4 = 0
        self.contador5 = 0
        self.contador6 = 0
        self.contadorVerde = 0
        self.contador9 = 0
        self.bandeira = False
        self.bandeira2 = False
        self.bandeira3 = False
        self.bandeira4 = False
        self.bandeira5 = False
        self.contours = []
        self.contours2 = []
        self.arm = True
        self.err = 0.0
        self.yaw = 0
        
        self.flag_corrigir_erro = False
        

        # Subscribers
        self.bridge = CvBridge()
        # self.image_sub = rospy.Subscriber('/camera/color/image_raw/compressed',CompressedImage,self.image_callback,queue_size=1,buff_size = 2**24) # CAMERA ANTIGA
        self.image_sub = rospy.Subscriber('/camera/image/compressed',CompressedImage,self.image_callback,queue_size=1,buff_size = 2**24) # CAMERA NOVA
        self.odom_sub = rospy.Subscriber("/odom",Odometry,self.odom_callback)
        self.laser_subscriber = rospy.Subscriber('/scan',LaserScan, self.laser_callback)

        # Publishers
        self.image_pub = rospy.Publisher("/image_publisher/", Image, queue_size=1)

        self.point_pub = rospy.Publisher("/center_publisher/", Point, queue_size=1)
        self.braco = rospy.Publisher("/joint1_position_controller/command", Float64, queue_size=1)
        self.pinca = rospy.Publisher("/joint2_position_controller/command", Float64, queue_size=1)

        self.cmd_vel_pub = rospy.Publisher("cmd_vel", Twist, queue_size=3)
        self.cmd_vel_pub.publish(Twist())
        


    def odom_callback(self, data: Odometry):
        self.odom = data
        self.x = data.pose.pose.position.x
        self.y = data.pose.pose.position.y
        self.z = data.pose.pose.position.z
        if self.start == None:
            self.start = data.pose.pose.position

        
        orientation_list = [data.pose.pose.orientation.x,
                            data.pose.pose.orientation.y,
                            data.pose.pose.orientation.z,
                            data.pose.pose.orientation.w]

        self.roll, self.pitch, self.yaw = euler_from_quaternion(orientation_list)

        # convert yaw from [-pi, pi] to [0, 2pi]
        self.yaw = self.yaw % (2*np.pi)

    def get_angular_error(self):
        x = self.point.x - self.x
        y = self.point.y - self.y
        theta = np.arctan2(y,x)

        self.distance = np.sqrt(x*2 + y*2)
        self.err = np.rad2deg(theta - self.yaw)
        self.err = (self.err + 100) % 360 - 100
        self.twist.angular.z = self.err * self.kp
        

    def laser_callback(self, msg: LaserScan) -> None:
        self.laser_msg = np.array(msg.ranges).round(decimals=2) # Converte para np.array e arredonda para 2 casas decimais
        self.laser_msg[self.laser_msg == 0] = np.inf
        self.frente = list(self.laser_msg)[0:5] + list(self.laser_msg)[-5:]
        print(f'Leitura na frente: {self.laser_msg[0]}')
        print("Faixa valida: ", msg.range_min , " - ", msg.range_max )

    def image_callback(self, msg: CompressedImage) -> None:
        """
        Callback function for the image topic
        """
        try:
            cv_image = self.bridge.compressed_imgmsg_to_cv2(msg, "bgr8")
        except CvBridgeError as e:
            print(e)
        
        self.color_segmentation(cv_image) # Processamento da imagem
        #self.image_pub.publish(self.bridge.cv2_to_imgmsg(self.mask, "bgr8"))

        if self.point.x != -1:
            cv_image = cv2.circle(cv_image, (self.point.x,self.point.y), 5, (0,0,255), -1)

        self.image_pub.publish(self.bridge.cv2_to_imgmsg(cv_image, "bgr8"))

        self.w = cv_image.shape[1]
        self.cx = self.point.x

    def color_segmentation(self,bgr: np.ndarray) -> None:
        self.get_angular_error()
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, self.lower_hsv, self.upper_hsv)
        mask2 = mask.copy()
        mask = mask[int(float(mask.shape[0])/2):,:]
        # mask = mask[:,int(float(self.w *0.1)):]
        self.mask = mask.copy()
        

        elemento_estrut = np.ones([10,10])
        mask2 = cv2.morphologyEx(mask2, cv2.MORPH_OPEN, elemento_estrut)
        self.contours,_ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        if len(self.contours) > 0:
            cnt = max(self.contours, key=lambda x: cv2.contourArea(x))
            M = cv2.moments(cnt)
            self.point.x = int(M['m10']/M['m00'])
            self.point.y = int(M['m01']/M['m00'])
            cv2.circle(mask,(self.point.x,self.point.y),4,(255,255,0),1)
        else:
            self.point.x = -1

        mask2 = mask2[:,int(float(mask2.shape[0])/0.9):]
        
        self.contours2,_ = cv2.findContours(mask2, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        maskVerde = cv2.inRange(hsv, self.lower_hsvVerde, self.upper_hsvVerde)
        maskVerde = cv2.morphologyEx(maskVerde, cv2.MORPH_OPEN, self.kernel)
        maskVerde = cv2.morphologyEx(maskVerde, cv2.MORPH_CLOSE, self.kernel)
        elemento_estrut = np.ones([10,10])
        maskVerde = cv2.morphologyEx(maskVerde, cv2.MORPH_OPEN, elemento_estrut)
        self.contoursVerde,_ = cv2.findContours(maskVerde, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        # cv2.imshow("maskVerde",maskVerde)
        # cv2.imshow("mask2",mask2)
        # cv2.imshow("mask",mask)
        # cv2.waitKey(1)

        if len(self.contoursVerde) > 0:
            cnt = max(self.contoursVerde, key = lambda x: cv2.contourArea(x))
            M = cv2.moments(cnt)
            if self.startVerdex == None:
                self.startVerdex = self.pointVerdex
            self.pointVerdex = int(M['m10']/M['m00'])
            self.pointVerdey = int(M['m01']/M['m00'])


    def aproxima(self) -> None:
        if self.arm:
            self.braco.publish(-1.0)
            self.pinca.publish(0.0)
            self.arm = False
        self.twist.angular.z = 0
        self.twist.linear.x = 0.1
        self.contador +=1
        if (self.contador) > 1600:
            self.robot_state = "gira1"
        
        

    def aproxima2(self) -> None:
        if self.contador2 == 0 :
            self.contador2 = 1
            self.flag_corrigir_erro = True
            self.err_inicial = self.w/2 - self.point.x
            print(f'ERRO INICIAL {self.err_inicial}')

        
        err = (self.w/2) - self.point.x
        print(f"self.w - {self.w})")
        print(f"self.point.x - {self.point.x})")
        #print(f"abs err - {abs(self.err)}")
        print(f"abs err - {abs(err)}\n\n")
        self.twist.linear.x = 0.1
        # if abs(self.err) < 200.0:
        #     self.flag_corrigir_erro = False
        #     self.err_inicial = 0
        if abs(err) < 200.0:
            self.flag_corrigir_erro = False
            self.err_inicial = 0
        else:
            # if abs(self.err_inicial) > abs(self.err):
            #     self.twist.angular.z = (float(self.err)/self.kp)
            if abs(self.err_inicial) > abs(err):
                self.twist.angular.z = -(float(err)/self.kp) #DIREITA
            else:
                # self.twist.angular.z = -(float(self.err)/self.kp)
                self.twist.angular.z = (float(err)/1500) #ESQUERDA

        self.contador4 +=0.1
        print(self.contador4)
        if (self.contador4) > 1850.0:
            self.robot_state = "gira2"
                          
    def gira1(self) -> None:
        self.twist = Twist()

        self.twist.angular.z = -0.1
        self.contador3 +=1
        if (self.contador3) > 3000:
            self.robot_state = "aproxima2"
        


    def gira2(self) -> None:
        self.twist = Twist()

        self.twist.angular.z = -0.1
        self.contador5 +=1
        if (self.contador5) > 3400:
            self.robot_state = "aproximaVerde"
    

    def aproximaVerde(self) -> None:
        print(f"Verde: {self.startVerdex}")
        print(f"Robo: {self.x}")
        self.twist.angular.z = 0
        self.twist.linear.x = 0.1
        err = self.w/2 - self.pointVerdex
        self.twist.angular.z = (float(err)/self.kp)
        self.contadorVerde += 0.1
        print(f"Contador: {self.contadorVerde}")
        self.contadorVerde +=0.1

        if min(self.frente) < 0.19:
            self.robot_state = "derrube"
            

        # if (self.contadorVerde > 350.5):
        #     self.robot_state = "gira3"
    
    def derrube(self) -> None:
        self.twist.angular.z = 0
        self.twist.linear.x = 0
        self.cmd_vel_pub.publish(self.twist)
        self.braco.publish(1)
        rospy.sleep(1)
        self.robot_state = "gira3"
    
    def gira3(self) -> None:
        self.twist = Twist()
        self.twist.angular.z = -0.1
        self.contador6 +=1
        if (self.contador6) > 6640:
            self.robot_state = "aproxima3"
    
    def aproxima3(self) -> None:
        self.twist.angular.z = 0
        self.twist.linear.x = 0.1 
        err = (self.w/2) - self.point.x
        print(err)
        if self.contador9 == 0 :
            self.contador9 = 1
            self.flag_corrigir_erro = True
            self.err_inicial = self.w/2 - self.point.x
            print(f'ERRO INICIAL {self.err_inicial}')
        if abs(err) < 200.0:
            self.flag_corrigir_erro = False
            self.err_inicial = 0
        else:
            # if abs(self.err_inicial) > abs(self.err):
            #     self.twist.angular.z = (float(self.err)/self.kp)
            if abs(self.err_inicial) > abs(err):
                self.twist.angular.z = -(float(err)/self.kp) #DIREITA
            else:
                # self.twist.angular.z = -(float(self.err)/self.kp)
                self.twist.angular.z = (float(err)/self.kp) #ESQUERDA
        
        if (self.x - self.start.x < 0.2):
            self.robot_state = "para"
        
    
    def para(self) -> None:
        self.twist.angular.z = 0
        self.twist.linear.x = 0
        self.cmd_vel_pub.publish(self.twist)



    def control(self) -> None:
        '''
        This function is called at least at {self.rate} Hz.
        This function controls the robot.
        Não modifique esta função.
        '''
        self.twist = Twist()
        print(f'self.robot_state: {self.robot_state}')
        self.robot_machine[self.robot_state]()

        self.cmd_vel_pub.publish(self.twist)
        
        self.rate.sleep()


if __name__ == "__main__":
    rospy.init_node("roda_exemplo")
    control = Control()
    #pub = rospy.Publisher("cmd_vel", Twist, queue_size=3)

    try:
        while not rospy.is_shutdown():
            control.control()
            #vel = Twist(Vector3(v,0,0), Vector3(0,0,w))
            #pub.publish(vel)
            # rospy.sleep(2.0)
    except rospy.ROSInterruptException:
        print("Ocorreu uma exceção com o rospy")