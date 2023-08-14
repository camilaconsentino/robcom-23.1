#! /usr/bin/env python3
# -*- coding:utf-8 -*-

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

""" 
	roslaunch my_simulation encaixotado.launch
	rosrun pf-robcomp q2.py
"""

class Questao2():
	def __init__(self):

		self.rate = rospy.Rate(250) # 250 Hz
		
		#SUBSCRIBERS
		self.bridge = CvBridge()
		self.image_sub = rospy.Subscriber('/camera/image/compressed', CompressedImage, self.image_callback, queue_size=1, buff_size = 2**24)
		self.odom_sub = rospy.Subscriber('/odom', Odometry, self.odom_callback, queue_size=1)

		#PUBLISHERS
		self.cmd_vel_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=3)
		self.cmd_vel_pub.publish(Twist()) #se eu to fazendo isso aqui, nao preciso fzr no resto do codigo??

		#para a garra
		self.braco = rospy.Publisher("/joint1_position_controller/command", Float64, queue_size=1)
		#self.garra = rospy.Publisher("/joint2_position_controller/command", Float64, queue_size=1)	

		#ESTADOS
		self.state = 1
		self.selected_mod = None
		self.robot_state = "rotate"
		self.robot_machine = {
			"rotate": self.rotate,
			"center_on_coord": self.center_on_coord,
			"garra": self.garra,
			"para": self.para
		}

		#VARIAVEIS
		self.twist = Twist()
		self.kp = 100 #(ou 200, qnt maior, mais preciso)
		self.kernel = np.ones((5,5),np.uint8) #para fazer morphologyEx
		self.initial_position = 0
		self.contador = 0
		self.ordem_aparicao = []
		self.angulo_aparicao = []

	#CALLBACKS
	def odom_callback(self, data: Odometry):
		self.position = data.pose.pose.position

		if self.initial_position == 0:
			self.initial_position = self.position

		orientation_list = [data.pose.pose.orientation.x,
							data.pose.pose.orientation.y,
							data.pose.pose.orientation.z,
							data.pose.pose.orientation.w]

		self.roll, self.pitch, self.yaw = euler_from_quaternion(orientation_list)


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
		self.centro_img = (w//2, h//2) # corte que fiz pra ele olhar o centro da imagem integralmente
		
		#mobile net
		_,self.resultadosMB = detect(net, img, CONFIDENCE, COLORS, CLASSES)
		if len(self.resultadosMB) > 0:
			self.classeMB = self.resultadosMB[0][0]
			self.centroMB = (self.resultadosMB[0][2][0] + self.resultadosMB[0][3][0])/2 # CENTRO X

			if self.state == 1:
				if self.classeMB == "horse" or self.classeMB == "car" or self.classeMB == "bicycle":
					if self.classeMB not in self.ordem_aparicao:
						self.ordem_aparicao.append(self.classeMB)
						self.angulo_aparicao.append(self.yaw)
						print(f"{self.classeMB} apareceu no angulo {self.yaw}")

			elif self.state == 2:
				if self.classeMB == "car":
					self.twist.angular.z = 0
					self.robot_state = "center_on_coord"

			elif self.state ==3:
				if self.classeMB == "bicyle":
					self.twist.angular.z = 0
					self.robot_state = "center_on_coord" 
			
			elif self.state == 4:
				if self.classeMB == "horse":
					self.twist.angular.z = 0
					self.robot_state = "center_on_coord"


	def rotate(self):
		self.twist.angular.z=0.5

		if self.state == 1:
			print(f"CONTADOR: {self.contador}")
			if self.contador >= 3100:
				self.robot_state = "para"
			self.contador+=1
	
	def center_on_coord(self) -> None:
		err = self.centro_img[0] - self.centroMB #centro do aruco, do cnt do bloco, do mobile net...
		print(err)
		if err >= 3:
			self.twist.angular.z = float(err)/self.kp
		else:
			self.twist.angular.z = 0
			self.robot_state = "garra"

	def garra(self) -> None:
		self.braco.publish(1.5)
		rospy.sleep(0.5)
		self.braco.publish(-1.5)
		rospy.sleep(1)

		if self.state == 2:
			self.state = 3
			self.robot_state = "rotate"
		elif self.state == 3:
			self.state = 4
			self.robot_state = "rotate"
		elif self.state == 4:
			self.robot_state = "para"
	
	def para(self) -> None:
		self.twist.linear.x = 0
		self.twist.angular.z = 0

		if self.state == 1:
			self.state = 2
			self.robot_state = "rotate"


	def control(self) -> None:
		'''
		This function is called at least at {self.rate} Hz.
		This function controls the robot.
		'''
		
		self.twist = Twist()
		print(f'self.robot_state: {self.robot_state}')
		self.robot_machine[self.robot_state]()
		print(f'self.state: {self.state}')
		print(f'ordem_aparicao: {self.ordem_aparicao}')

		self.cmd_vel_pub.publish(self.twist)

		self.rate.sleep()


def main():
	rospy.init_node('q2')
	control = Questao2()
	rospy.sleep(1)

	while not rospy.is_shutdown():
		control.control()

if __name__=="__main__":
	main()