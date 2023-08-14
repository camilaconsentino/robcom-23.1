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

class Control():
	def __init__(self):
		self.rate = rospy.Rate(250) # 250 Hz

		# HSV Filter
		self.color_param = {
			"blue": {
				"lower": 225,
				"upper": 255
			},
			"yellow": {
				"lower": 45,
				"upper": 75
			},
		}

		# Subscribers
		self.bridge = CvBridge()
		self.image_sub = rospy.Subscriber('/camera/image/compressed', CompressedImage, self.image_callback, queue_size=1, buff_size = 2**24)
		self.odom_sub = rospy.Subscriber('/odom', Odometry, self.odom_callback, queue_size=1)

		#PUBLISHERS
		self.cmd_vel_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=3)
		self.cmd_vel_pub.publish(Twist()) #se eu to fazendo isso aqui, nao preciso fzr no resto do codigo??

		self.selected_color = None
		self.robot_state = "segue"
		self.robot_machine = {
			"segue": self.segue,
			"center_on_coord": self.center_on_coord,
			"para": self.para
		}

		self.twist = Twist()
		self.kp = 200 #(ou 200, qnt maior, mais preciso)
		self.kernel = np.ones((5,5),np.uint8)
	
	def color_segmentation(self, hsv: np.ndarray, lower, upper) -> Point:
		""" 
		Use HSV color space to segment the image and find the center of the object.

		Args:
			bgr (np.ndarray): image in BGR format
		
		Returns:
			Point: x, y and area of the object
		"""
		
		self.lower_hsv = np.array([lower//2,50,50],dtype=np.uint8)
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

		#return: centro, mask, area
		return point, existencia
		
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
		self.centro_segue = (h, 25*h//40) # corte que eu fiz só para que ele olhe para a pista
		self.centro_img = (w//2, h//2) # corte que fiz pra ele olhar o centro da imagem integralmente

		# color segmentation
		self.centro_yellow, _ = self.color_segmentation(hsv, 45, 75)
		self.centro_blue, _ = self.color_segmentation(hsv, 45, 75)

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

		if self.err <= 1.5:
			self.twist.linear.x = 0.5
		else:
			self.twist.linear.x = 0.3

		if self.position.x < -4 and self.position.y < 4:
			self.robot_state = "para"

	def para(self) -> None:
		"""
		Stop the robot
		"""
		self.twist.linear.x = 0

	def center_on_coord(self):
		self.err = self.centro_img[0] - self.centro_yellow[0] #centro do contorno/ da mask da pista
		self.twist.angular.z = float(self.err)/self.kp

		print(f"CENTRO IMAGEM: {self.centro_segue[0]}")
		print(f'CENTRO MASK: {self.centro_yellow[0]}')
		print(f'ERRO: {self.err}')

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
