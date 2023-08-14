#! /usr/bin/env python3
# -*- coding:utf-8 -*-

#IMPORTS
import numpy as np
from numpy import random
from math import asin
import math
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

class Control():
	def __init__(self):
		self.rate = rospy.Rate(250) # 250 Hz

		self.contador = 0
		# Subscribers
		self.bridge = CvBridge()
		self.laser_subscriber = rospy.Subscriber('/scan',LaserScan, self.laser_callback)
		self.image_sub = rospy.Subscriber('/camera/image/compressed', CompressedImage, self.image_callback, queue_size=1, buff_size = 2**24)
		self.odom_sub = rospy.Subscriber('/odom', Odometry, self.odom_callback, queue_size=1)

		# Publishers
		self.cmd_vel_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=3)
		self.cmd_vel_pub.publish(Twist())

		self.state = 1
		self.selected_mod = None
		self.robot_state = "rotate"
		self.robot_machine = {
			"rotate": self.rotate,
			"aproxima": self.aproxima,
			"center_on_coord": self.center_on_coord,
			"para": self.para,
			"anda": self.anda
		}

		#VARIAVEIS
		self.twist = Twist()
		self.kp = 200 #qnt maior, mais preciso
		self.kernel = np.ones((5,5),np.uint8) #para fazer morphologyEx
		self.contador = 0
	
	def laser_callback(self, msg: LaserScan) -> None:
		self.laser_msg = np.array(msg.ranges).round(decimals=2)
		self.laser_msg[self.laser_msg == 0] = np.inf

		self.laser_forward = np.min(list(self.laser_msg[0:5]) + list(self.laser_msg[354:359])) 
		self.laser_backwards = np.min(list(self.laser_msg[175:185]))
		
	def image_callback(self, msg: CompressedImage) -> None:
		"""
		Callback function for the image topic
		"""
		try:
			cv_image = self.bridge.compressed_imgmsg_to_cv2(msg, "bgr8")
			hsv = cv2.cvtColor(cv_image, cv2.COLOR_BGR2HSV)
		except CvBridgeError as e:
			print(e)
		
		#dimensao
		h, w, d = cv_image.shape
		self.centro_img = (w//2, h//2) # corte que fiz pra ele olhar o centro da imagem integralmente

		#mobile net
		_,self.resultadosMB = detect(net, cv_image, CONFIDENCE, COLORS, CLASSES)

		if len(self.resultadosMB)!=0:
			self.classeMB = self.resultadosMB[0][0]
			self.centroMB = (self.resultadosMB[0][2][0] + self.resultadosMB[0][3][0])/2
			
			if self.state == 1 and self.classeMB == "cow":
				self.robot_state = "para"

			if self.state == 3 and self.classeMB == "car":
				self.robot_state = "aproxima"

	def odom_callback(self, data: Odometry):
		self.position = data.pose.pose.position
		
		orientation_list = [data.pose.pose.orientation.x,
							data.pose.pose.orientation.y,
							data.pose.pose.orientation.z,
							data.pose.pose.orientation.w]

		self.roll, self.pitch, self.yaw = euler_from_quaternion(orientation_list)

		self.yaw = self.yaw % (2*np.pi)


	def rotate(self) -> None:
		"""
		Rotate the robot
		"""
		if self.state == 1:	
			self.twist.angular.z = 0.5
		elif self.state == 3:
			if self.contador <= 1800:
				self.twist.angular.z = 0.5
			else:
				self.twist.angular.z = 0
				self.robot_state = "anda"
			self.contador += 1
			print(f"CONTADOR: {self.contador}")
			

	def aproxima(self) -> None:
		self.center_on_coord()
		if self.laser_forward >= 0.3:
			self.twist.linear.x = 0.3
		else:
			self.robot_state = "para"

	def para(self) -> None:
		if self.state == 1:
			self.twist.angular.z = 0.0
			self.state = 2
			self.robot_state = "aproxima"

		elif self.state == 2:
			self.twist.linear.x = 0.0
			self.twist.angular.z = 0
			self.state = 3
			self.robot_state = "rotate"

		else:
			self.twist.angular.z = 0.0
			self.twist.linear.x = 0.0
		
	def center_on_coord(self):
		print(self.classeMB)
		print(f'CENTRO IMG: {self.centro_img[0]}')
		print(f'CENTRO MB: {self.centroMB}')
		err = self.centro_img[0] - self.centroMB #centro X do aruco, do cnt do bloco, do mobile net...
		self.twist.angular.z = float(err)/self.kp

	def anda(self) -> None:
		print("ANDA")
		self.twist.linear.x = 0.2
		
	def control(self):
		'''
		This function is called at least at {self.rate} Hz.
		This function controls the robot.
		'''
		self.twist = Twist()
		print(f'self.robot_state: {self.robot_state}')
		print(f'self.state: {self.state}')
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
