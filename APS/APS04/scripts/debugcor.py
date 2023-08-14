#! /usr/bin/env python3
# -- coding:utf-8 --

import rospy

import numpy as np
import cv2
from geometry_msgs.msg import Twist, Vector3
from geometry_msgs.msg import Twist, PointStamped, Point
from sensor_msgs.msg import LaserScan
from sensor_msgs.msg import CompressedImage, Image
from cv_bridge import CvBridge,CvBridgeError
import numpy as np

""" 
Running
	roslaunch my_simulation pista_s2.launch
	rosrun aps4 cor.py
"""

class Control():
	def _init_(self):
		self.rate = rospy.Rate(250) # 250 Hz
		
		self.cx = -1
		self.kp = 100
		self.w = 0
		
		self.robot_state = "procura"
		self.robot_machine = {
			"procura": self.procura,
			"aproxima": self.aproxima,
			"para": self.para
		}

		# HSV Filter
		self.lower_hsv = np.array([100,60,60],dtype=np.uint8) # Blue
		self.upper_hsv = np.array([140,255,255],dtype=np.uint8)
		self.kernel = np.ones((5,5),np.uint8)

		# Image
		self.point = Point()
		self.point.x = -1
		self.point.y = -1

		self.dormir = 1

		# Subscribers
			#Imagem
		self.bridge = CvBridge()
		self.image_sub = rospy.Subscriber('/camera/image/compressed', CompressedImage, self.image_callback, queue_size=1, buff_size = 2**24)
			#Laser
		self.laser_subscriber = rospy.Subscriber('/scan',LaserScan, self.laser_callback)

		# Publishers
			#Movimentacao
		self.cmd_vel_pub = rospy.Publisher("cmd_vel", Twist, queue_size=3)
		self.cmd_vel_pub.publish(Twist()) #Caso ele esteja andando reseta ele e deixa ele parado

			#Imagem
		self.image_pub = rospy.Publisher('image_publisher', Image, queue_size=1)
		self.point_pub = rospy.Publisher('center_publisher', Point, queue_size=1)

		
	def laser_callback(self, msg: LaserScan) -> None:
		self.laser_msg = np.array(msg.ranges).round(decimals=2) # Converte para np.array e arredonda para 2 casas decimais
		self.laser_msg[self.laser_msg == 0] = np.inf
	
	def image_callback(self, msg: CompressedImage) -> None:
		"""
		Callback function for the image topic
		"""
		try:
			cv_image = self.bridge.compressed_imgmsg_to_cv2(msg, "bgr8")
		except CvBridgeError as e:
			print(e)
		
		self.color_segmentation(cv_image) #Processa a imagem
		if self.point.x != -1 and self.point.y != -1:
			cv2.circle(cv_image, (int(self.point.x), int(self.point.y)), 5, (0, 0, 255), -1) #Desenha o centro do creeper

		self.w = cv_image.shape[1]
		self.cx = self.point.x

		self.image_pub.publish(self.bridge.cv2_to_imgmsg(cv_image, "bgr8"))
		

	def color_segmentation(self,bgr: np.ndarray) -> None:
		""" 
		Use HSV color space to segment the image and find the center of the object.

		Args:
			bgr (np.ndarray): image in BGR format
		"""
		self.point = Point() # Reseta o ponto
		hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
		mask = cv2.inRange(hsv,self.lower_hsv,self.upper_hsv)
		mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self.kernel) #Limpa interferencias externas
		mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self.kernel) #Limpa buracos internos, apos remover as interferencias externas
		cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
		
		if len(cnts) > 0:
			cnt = max(cnts, key=lambda x: cv2.contourArea(x)) #Pega o maior contorno
			M = cv2.moments(cnt)
			self.point.x = int(M["m10"] / M["m00"])
			self.point.y = int(M["m01"] / M["m00"])
		else:
			self.point.x = -1
			self.point.y = -1

	def procura(self) -> None:
		"""
		Find the creeper
		"""
		#Gira ate achar o creeper
		if self.point.x == -1 and self.point.y == -1:
			#self.color_segmentation(bgr)
			vel = Twist()
			vel.linear.x = 0
			vel.angular.z = 0.6
			self.cmd_vel_pub.publish(vel)
		else:
			self.robot_state = "aproxima"
		

		

	def aproxima(self) -> None:
		"""
		Go to the creeper
		"""
		# Controla o robo para ficar alinhado com o creeper
		if self.laser_msg[0] >= 0.3:
			err = self.w/2 - self.cx
			self.twist.angular.z = float(err) / self.kp
			self.twist.linear.x = 0.1
		else:
			self.robot_state = "para"

	
	def para(self) -> None:
		"""
		Stop the robot
		"""
		vel = Twist()
		vel.linear.x = 0
		vel.angular.z = 0
		self.cmd_vel_pub.publish(vel)


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

def main():
	rospy.init_node('Controler')
	control = Control()
	rospy.sleep(1)

	while not rospy.is_shutdown():
		control.control()

if __name__=="main_":
	main()