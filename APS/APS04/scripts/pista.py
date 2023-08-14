#! /usr/bin/env python3
# -*- coding:utf-8 -*-

import rospy
import cv2
import numpy as np

from geometry_msgs.msg import Twist, Vector3
from geometry_msgs.msg import Point, Twist, PointStamped
from sensor_msgs.msg import Image, CompressedImage
from cv_bridge import CvBridge,CvBridgeError

""" 
Running
	roslaunch my_simulation trevo.launch
	rosrun aps4 pista.py
"""

class Control():
	def __init__(self):
		#super().__init__()

		self.rate = rospy.Rate(250) # 250 Hz
		

		self.robot_state = "checa"
		self.robot_machine = {
			"checa": self.checa,
			"anda": self.anda,
			"para": self.para
		}

		#SUBSCRIBERS
		self.bridge = CvBridge()
		#self.image_sub = rospy.Subscriber("/image_publisher/", Image,self.image_callback,queue_size=1,buff_size = 2**24)
		self.image_sub = rospy.Subscriber('/camera/image/compressed',CompressedImage,self.image_callback,queue_size=1,buff_size = 2**24)
		self.point_sub = rospy.Subscriber("/center_publisher/", Point,self.image_callback,queue_size=1,buff_size = 2**24)

		#PUBLISHERS
		self.cmd_vel_pub = rospy.Publisher("/cmd_vel", Twist, queue_size=3)
		self.cmd_vel_pub.publish(Twist())
		self.point_pub = rospy.Publisher('center_publisher', Point, queue_size=1)
		self.image_pub = rospy.Publisher('image_publisher', Point, queue_size=1)
		
		self.point = Point(-1,-1,-1)
		self.image = None

		self.vel = Twist()

	def image_callback(self, msg: CompressedImage) -> None:
		"""
		Callback function for the image topic
		"""
		try:
			cv_image = self.bridge.compressed_imgmsg_to_cv2(msg, "bgr8")
		except CvBridgeError as e:
			print(e)

		# FAZ VISAO AQUI
		self.color_segmentation(cv_image)
		self.robot_state = 'checa'

		self.image = cv_image
		self.image_pub.publish(self.bridge.cv2_to_imgmsg(cv_image, "bgr8"))
	
	def color_segmentation(self,bgr: np.ndarray) -> None:
		""" 
		Use HSV color space to segment the image and find the center of the object.

		Args:
			bgr (np.ndarray): image in BGR format
		"""
		print('entrei')
		
		self.lower_hsv = np.array([45//2,60,60],dtype=np.uint8) #yellow
		self.upper_hsv = np.array([75//2,255,255],dtype=np.uint8)
		hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
		mask = cv2.inRange(hsv, self.lower_hsv, self.upper_hsv)

		self.kernel = np.ones((5,5),np.uint8)
		mask = cv2.morphologyEx(mask,cv2.MORPH_OPEN, self.kernel)
		mask = cv2.morphologyEx(mask,cv2.MORPH_CLOSE, self.kernel)

		# find contours
		contours,_ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

		print(contours)
		
		if len(contours) > 0:
			# find contour with max area
			cnt = max(contours, key = lambda x: cv2.contourArea(x))

			# Find the center
			M = cv2.moments(cnt)
			self.point.x = int(M['m10']/M['m00'])
			self.point.y = int(M['m01']/M['m00'])
			
			print("tem contorno")

		else:
			self.point.x = -1
			self.point.y = -1
	
	def checa(self) -> None:

		if self.point.x != -1 or self.point.y != -1:
			self.robot_state = 'anda'
		else:
			self.robot_state = 'para'

	def anda(self) -> None:
		
		#segue a faixa
		
		self.kp = 200
		self.w = self.image.shape[1]
			
		erro = self.w/2 - self.point.x
		self.twist.angular.z = float(erro) / self.kp 
		self.twist.linear.x = 0.2

		self.cmd_vel_pub.publish(self.vel)

		self.robot_state = 'anda'

	def para(self) -> None:
		self.twist.linear.x = 0
		self.cmd_vel_pub.publish(self.vel)
		self.robot_state = 'checa'

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

if __name__=="__main__":
	main()