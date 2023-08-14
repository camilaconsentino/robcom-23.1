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

""" 
	roslaunch my_simulation circuito.launch
	rosrun pf-robcomp q1.py
"""

class Questao1():
	def __init__(self, cor_caixa):
		self.rate = rospy.Rate(250) # 250 Hz
		self.cor_caixa = cor_caixa
		
		#SUBSCRIBERS
		self.bridge = CvBridge()
		self.laser_subscriber = rospy.Subscriber('/scan',LaserScan, self.laser_callback)
		self.image_sub = rospy.Subscriber('/camera/image/compressed', CompressedImage, self.image_callback, queue_size=1, buff_size = 2**24)
		self.odom_sub = rospy.Subscriber('/odom', Odometry, self.odom_callback, queue_size=1)

		#PUBLISHERS
		self.cmd_vel_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=3)
		self.cmd_vel_pub.publish(Twist()) #se eu to fazendo isso aqui, nao preciso fzr no resto do codigo??

		#ESTADOS
		self.state = 1
		self.selected_mod = None
		self.robot_state = "segue"
		self.robot_machine = {
			"segue": self.segue,
			"rotate": self.rotate,
			"aproxima": self.aproxima,
			"go_to_coord": self.go_to_coord,
			"para": self.para
		}

		#VARIAVEIS
		self.twist = Twist()
		self.kp = 100 #(ou 200, qnt maior, mais preciso)
		self.kernel = np.ones((5,5),np.uint8) #para fazer morphologyEx
		self.initial_position = 0
		self.contador = 0
		self.contadorR = 0
	
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

		cv2.imshow('Mask', mask)
		cv2.waitKey(1)   

		return point, existencia
	
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
		self.centro_yellow, self.existenciaY = self.color_segmentation(hsv, 45, 75)
		self.centro_blue, self.existenciaB = self.color_segmentation(hsv, 225, 265)
		self.centro_red, self.existenciaR = self.color_segmentation(hsv, 0, 0)
		self.centro_green, self.existenciaG = self.color_segmentation(hsv, 105, 135)

		#if self.state == 2:
		#	if self.existenciaB and self.existenciaG and self.existenciaR:
		#		self.state = "para"

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

	def segue(self) -> None:
		self.center_on_coord()
		
		self.difX = abs(self.position.x - self.initial_position.x)
		self.difY = abs(self.position.y - self.initial_position.y)
		print(f'Diferenca entre posicao atual e inicial: X {self.difX}, Y {self.difY}')

		self.twist.linear.x = 0.5

		if self.contador > 1000:
			if self.difX < 0.3 and self.difY < 0.3:
				self.robot_state = "para"

		print(self.contador)
		self.contador+=1

	def rotate(self) -> None:
		if self.contadorR <= 890:
			self.twist.angular.z = 0.5
		else:
			self.robot_state = "para"
		self.contadorR+=1
		print(f'CONTADOR: {self.contadorR}')
	
	def aproxima(self) -> None:
		self.center_on_coord()

		if self.laser_forward > 0.2:
			self.twist.linear.x = 0.4
		else:
			self.robot_state = "para"
	
	def center_on_coord(self) -> None:
		if self.state == 1:
			self.err = self.centro_img[0] - self.centro_yellow[0] 
		if self.state == 3:
			if self.cor_caixa == "verde":	
				self.err = self.centro_img[0] - self.centro_green[0]
			elif self.cor_caixa == "azul":
				self.err = self.centro_img[0] - self.centro_blue[0]
			else:
				self.err = self.centro_img[0] - self.centro_red[0]

		self.twist.angular.z = float(self.err)/self.kp
	
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

	def para(self) -> None:
		self.twist.linear.x = 0
		self.twist.angular.z = 0

		if self.state == 1:
			self.state = 2
			self.robot_state = "rotate"
		elif self.state == 2:
			self.state = 3
			self.robot_state = "aproxima"
		elif self.state == 3:
			self.state = 4
			self.robot_state = "go_to_coord"
	
	def control(self) -> None:
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
	rospy.init_node('q1')
	# use as linhas abaixo para testar seu programa
	control = Questao1('verde')
	#control = Questao1('vermelha')
	#control = Questao1('azul')
	rospy.sleep(1)

	while not rospy.is_shutdown():
		control.control()

if __name__=="__main__":
	main()