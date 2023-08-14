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
import cv2.aruco as aruco

""" 
	roslaunch my_simulation reuniao.launch
	rosrun pf-robcomp q3.py
"""

class Questao3():
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
		self.robot_state = "rotate"
		self.robot_machine = {
			"rotate": self.rotate,
			"aproxima": self.aproxima,
			"avanca": self.avanca,
			"para": self.para,
			"fim": self.fim
		}

		#VARIAVEIS
		self.twist = Twist()
		self.kp = 100 #(ou 200, qnt maior, mais preciso)
		self.kernel = np.ones((5,5),np.uint8) #para fazer morphologyEx
		self.initial_position = 0
		self.contador = 0

		#ARUCO:
		#camera 
		self.camera_distortion = np.loadtxt('aruco/cameraDistortion_realsense.txt', delimiter=',')
		self.camera_matrix = np.loadtxt('aruco/cameraMatrix_realsense.txt', delimiter=',')

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
		self.centro_img = (w/2, h//2) # corte que fiz pra ele olhar o centro da imagem integralmente

		# color segmentation
		self.centro_yellow, self.existenciaY = self.color_segmentation(hsv, 45, 75)
		self.centro_blue, self.existenciaB = self.color_segmentation(hsv, 165, 195)
		self.centro_green, self.existenciaG = self.color_segmentation(hsv, 105, 135)

		#aruco
		self.ids, self.centros, self.distancia_aruco = self.geraAruco(cv_image)

		if self.state == 1:
			if len(self.ids[0]) == 1:
				self.robot_state = "para"

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

	def rotate(self):
		if self.contador == 6:
			self.robot_state = "fim"
		self.twist.angular.z = 0.5
            
	def center_on_coord(self) -> None:       
		err = self.centro_img[0] - self.centros[0][0] #centro do aruco, do cnt do bloco, do mobile net...

		self.twist.angular.z = float(err)/self.kp

	def aproxima(self) -> None:
		self.center_on_coord()

		if self.laser_forward >= 0.3:
			self.twist.linear.x = 0.2
		else:
			self.twist.angular.z = 0
			self.twist.linear.x = 0
			self.robot_state = "avanca"

	def avanca(self) -> None:
		if self.existenciaY or self.existenciaG or self.existenciaB:
			self.twist.linear.x = 0.2
		else:
			self.robot_state = "para"

	def para(self) -> None:
		if self.state == 1:
			self.state = 2
			self.twist.linear.x = 0
			self.twist.angular.z = 0
			self.robot_state = "aproxima"
		elif self.state == 2:
			self.contador += 1
			self.state = 1
			self.twist.linear.x = 0
			self.twist.linear.z = 0
			self.robot_state = "rotate"

	def fim(self) -> None:
		self.twist.linear.x = 0
		self.twist.linear.z = 0
		print("FIM")

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
	rospy.init_node('q3')
	control = Questao3()
	rospy.sleep(1)

	while not rospy.is_shutdown():
		control.control()

if __name__=="__main__":
	main()