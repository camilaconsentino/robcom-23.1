#! /usr/bin/env python3
# -*- coding:utf-8 -*-

""" 
Running
    roslaunch my_simulation pista23-1.launch
    rosrun aps4 pista.py
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
            "gira2" : self.gira2,
            "para" : self.para
        }

        # HSV Filter
        # self.lower_hsv = np.array([22, 41, 231],dtype=np.uint8)
        # self.upper_hsv = np.array([40, 255, 255],dtype=np.uint8)

        # self.lower_hsv2 = np.array([0, 30, 196],dtype=np.uint8)
        # self.upper_hsv2 = np.array([50, 153, 255],dtype=np.uint8)

        # self.lower_hsv3 = np.array([29, 30, 207],dtype=np.uint8)
        # self.upper_hsv3 = np.array([54, 146, 255],dtype=np.uint8)

        # self.lower_hsv4 = np.array([8, 113, 188],dtype=np.uint8)
        # self.upper_hsv4 = np.array([44, 159, 255],dtype=np.uint8)

        self.lower_hsv = np.array([29, 67, 174],dtype=np.uint8)
        self.upper_hsv = np.array([65, 172, 255],dtype=np.uint8)


        self.lower_hsvVerde = np.array([78, 71, 45],dtype=np.uint8)
        self.upper_hsvVerde = np.array([90, 255, 177],dtype=np.uint8)
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

        self.kp = 300
        self.w = 0
        self.cx = -1
        self.existencia = False
        self.contador = 0
        self.bandeira = False
        self.bandeira2 = False
        self.bandeira3 = False
        self.bandeira4 = False
        self.contours = []
        self.contours2 = []

        # Subscribers
        self.bridge = CvBridge()
        self.image_sub = rospy.Subscriber('/camera/color/image_raw/compressed',CompressedImage,self.image_callback,queue_size=1,buff_size = 2**24)
        self.odom_sub = rospy.Subscriber("/odom",Odometry,self.odom_callback)
        self.laser_subscriber = rospy.Subscriber('/scan',LaserScan, self.laser_callback)

        # Publishers
        self.image_pub = rospy.Publisher("/image_publisher/", Image, queue_size=1)
        self.point_pub = rospy.Publisher("/center_publisher/", Point, queue_size=1)

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

        if self.point.x != -1:
            cv_image = cv2.circle(cv_image, (self.point.x,self.point.y), 5, (0,0,255), -1)

        self.image_pub.publish(self.bridge.cv2_to_imgmsg(cv_image, "bgr8"))

        self.w = cv_image.shape[1]
        self.cx = self.point.x

    def color_segmentation(self,bgr: np.ndarray) -> None:
        self.get_angular_error()
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, self.lower_hsv, self.upper_hsv)
        #mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self.kernel)
        #mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self.kernel)
        # mask3 = cv2.inRange(hsv, self.lower_hsv2, self.upper_hsv2)
        # mask3 = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self.kernel)
        # mask3 = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self.kernel)
        # mask4 = cv2.inRange(hsv, self.lower_hsv3, self.upper_hsv3)
        # mask5 = cv2.inRange(hsv, self.lower_hsv4, self.upper_hsv4)
        # mask = cv2.bitwise_or(mask, mask3, mask=mask)
        # mask = cv2.bitwise_or(mask, mask4, mask=mask)
        # mask = cv2.bitwise_or(mask, mask5, mask=mask)
        mask2 = mask.copy()
        mask = mask[int(float(mask.shape[0])/2):,:]

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

        mask2 = mask2[:,int(float(mask2.shape[0])/0.9):]
        
        self.contours2,_ = cv2.findContours(mask2, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        # if len(self.contours2) > 0:
        #     cnt = max(self.contours2, key = lambda x: cv2.contourArea(x))
        #     self.existencia = True
        # else:
        #     if self.existencia == True and len(self.contoursVerde)==0 and self.bandeira3 == False:
        #         if self.robot_state == "aproxima" or self.robot_state == "segue":
        #             self.robot_state = "aproxima2"
        #             print('\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n foi')
        #             self.existencia = False

        maskVerde = cv2.inRange(hsv, self.lower_hsvVerde, self.upper_hsvVerde)
        maskVerde = cv2.morphologyEx(maskVerde, cv2.MORPH_OPEN, self.kernel)
        maskVerde = cv2.morphologyEx(maskVerde, cv2.MORPH_CLOSE, self.kernel)
        elemento_estrut = np.ones([10,10])
        maskVerde = cv2.morphologyEx(maskVerde, cv2.MORPH_OPEN, elemento_estrut)
        self.contoursVerde,_ = cv2.findContours(maskVerde, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        # cv2.imshow("maskVerde",maskVerde)
        cv2.imshow("mask2",mask2)
        cv2.imshow("mask",mask)
        cv2.waitKey(1)
        if len(self.contoursVerde) > 0:
            cnt = max(self.contoursVerde, key = lambda x: cv2.contourArea(x))
            M = cv2.moments(cnt)
            self.pointVerdex = int(M['m10']/M['m00'])
            self.pointVerdey = int(M['m01']/M['m00'])


    def aproxima(self) -> None:
        self.twist.angular.z = 0
        self.twist.linear.x = 0.1
        if len(self.contours2) > 0:
            cnt = max(self.contours2, key = lambda x: cv2.contourArea(x))
            self.existencia = True
        else:
            if self.existencia == True and len(self.contoursVerde)==0 and self.bandeira3 == False:
                if self.robot_state == "aproxima" or self.robot_state == "segue":
                    self.robot_state = "aproxima2"
                    print('\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n foi')
                    self.existencia = False
        
        

    def aproxima2(self) -> None:
        self.contador +=0.1
        print(self.contador)
        if self.contador == 45:
            self.robot_state = "gira1"
        self.twist.angular.z = 0
        self.twist.linear.x = 0.1
        err = self.w/2 - self.point.x
        self.twist.angular.z = float(err)/self.kp
        if self.x > (self.start.x + 5) and self.bandeira == True:
            self.robot_state = "gira1"
        
        if (self.x  < 0.2 and self.y < 0.2) and len(self.contours) == 0 and len(self.contours2) == 0:
            self.robot_state = "para"
            
    
    def gira1(self) -> None:
        self.twist = Twist()

        self.twist.angular.z = -0.05
        self.cmd_vel_pub.publish(self.twist)
        errox = abs(abs(self.start.x) - abs(self.x))
        if len(self.contours) > 0 and len(self.contours2) > 0:
            print(self.contours)
            print(self.contours2)
            print('\n\n\n\n\n\n\n')
            self.robot_state = "aproxima2"

        if len(self.contours2) > 0 and len(self.contours) < 2:
            self.robot_state = "segue"
        if len(self.contours2) == 0 and len(self.contoursVerde) == 0 and self.x > (self.start.x + 3):
            self.robot_state = "segue"

    def gira2(self) -> None:
        self.twist = Twist()
        self.twist.angular.z = -0.2
        self.cmd_vel_pub.publish(self.twist)
        
        if len(self.contours2) > 0:
            self.robot_state = "segue"

    
    def segue(self) -> None:
        self.bandeira = True
        self.twist.angular.z = 0
        self.twist.linear.x = 0.05
        err = self.w/2 - self.point.x
        self.twist.angular.z = (float(err)/self.kp)
        if self.x > 1:
            self.bandeira3 == False
        else:
            self.bandeira3 == True
        if len(self.contoursVerde) > 0 and self.x> (self.start.x + 5) and self.bandeira3 == False:
            self.robot_state = "aproximaVerde"
        
        

    def aproximaVerde(self) -> None:
        self.twist.angular.z = 0
        self.twist.linear.x = 0.1
        err = self.w/2 - self.pointVerdex
        self.twist.angular.z = (float(err)/self.kp)
        
        errox = (abs(self.pointVerdex) - abs(self.point.x))
        erroy = (abs(self.pointVerdey) - abs(self.point.y))
        if  erroy < 0 and errox < 0:
            self.robot_state = "gira2"
        
    
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