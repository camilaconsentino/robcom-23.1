#! /usr/bin/env python3
# -*- coding:utf-8 -*-

import rospy
import random
import cv2
import numpy as np
from cv_bridge import CvBridge,CvBridgeError
from geometry_msgs.msg import Point, Twist, PointStamped
from sensor_msgs.msg import LaserScan
from sensor_msgs.msg import Image, CompressedImage
from nav_msgs.msg import Odometry

""" 
Running
	roslaunch my_simulation caixas.launch
	rosrun aps4 aleatorio.py
"""

class Control():
	def __init__(self):
		self.rate = rospy.Rate(250) # 250 Hz

		# HSV Filter
		self.color_param = {
			"magenta": {
				"lower": 285//2,
				"upper": 315//2
			},
			"yellow": {
				"lower": 45//2,
				"upper": 75//2
			},
		}

		self.time = random.uniform(0.2, 2.0)
		self.target_time = 0

		# Subscribers
		self.bridge = CvBridge()
		self.odom_sub = rospy.Subscriber('/nav_msgs', Odometry, self.odom_callback)
		self.laser_subscriber = rospy.Subscriber('/scan',LaserScan, self.laser_callback)
		self.image_sub = rospy.Subscriber("/image_publisher/", Image,self.image_callback,queue_size=1,buff_size = 2**24)
		
		# Publishers
		self.cmd_vel_pub = rospy.Publisher()

		self.cmd_vel_pub.publish(Twist())

		self.selected_color = None
		self.robot_state = "rotate"
		self.robot_machine = {
			"rotate": self.rotate,
			"checar": self.checar,
			"center_on_coord": self.center_on_coord,
			"go_to_coord": self.go_to_coord,
			"para": self.para,
		}

		self.magenta_machine = {
			"aproxima": self.aproxima,
		}

		self.yellow_machine = {
			"afasta": self.afasta,
		}

	def odom_callback(self, data: Odometry):
		self.position = data.pose.pose.position
		
		orientation_list = [data.pose.pose.orientation.x,
							data.pose.pose.orientation.y,
							data.pose.pose.orientation.z,
							data.pose.pose.orientation.w]

		self.roll, self.pitch, self.yaw = euler_from_quaternion(orientation_list)

		self.yaw = self.yaw % (2*np.pi)
	
	def laser_callback(self, msg: LaserScan) -> None:
		self.laser_msg = np.array(msg.ranges).round(decimals=2)
		self.laser_msg[self.laser_msg == 0] = np.inf

		self.laser_forward = self.laser_msg[0]
		self.laser_backwards = self.laser_msg[180]
	
	def color_segmentation(self, bgr: np.ndarray, lower_hsv: np.ndarray, upper_hsv: np.ndarray,) -> Point:
		""" 
		Use HSV color space to segment the image and find the center of the object.

		Args:
			bgr (np.ndarray): image in BGR format
		
		Returns:
			Point: x, y and area of the object
		"""

		hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
		self.kernel = np.ones((5,5),np.uint8)

		mask = cv2.inRange(hsv, lower_hsv, upper_hsv)
		mask = cv2.morphologyEx(mask,cv2.MORPH_OPEN, self.kernel)
		mask = cv2.morphologyEx(mask,cv2.MORPH_CLOSE, self.kernel)

		contours,_ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE) 

		if len(contours) > 0:
			# find contour with max area
			cnt = max(contours, key = lambda x: cv2.contourArea(x))

			# Find the center
			M = cv2.moments(cnt)
			self.point.x = int(M['m10']/M['m00'])
			self.point.y = int(M['m01']/M['m00'])

			#area
			ct = sorted(contours, key=cv2.contourArea)[-1:]
			area = cv2.contourArea(ct)

		else:
			self.point.x = -1
			self.point.y = -1
			area = 0

		return self.point.x, self.point.y, area
		
	def image_callback(self, msg: CompressedImage) -> None:
		"""
		Callback function for the image topic
		"""
		try:
			cv_image = self.bridge.compressed_imgmsg_to_cv2(msg, "bgr8")
			hsv = cv2.cvtColor(cv_image, cv2.COLOR_BGR2HSV)
		except CvBridgeError as e:
			print(e)

		self.image = cv_image
		
		...

	def rotate(self) -> None:
		"""
		Rotate the robot
		"""
		self.temporodando = rospy.Time.now().to_sec()

		while self.temporodando < self.time:	
			self.twist.angular.z = 0.5
		
		self.robot_state = 'para'

	def checar(self) -> None:
		"""
		Stop the robot
		"""
		self.point.xy, self.point.yy, self.areaY = self.color_segmentation(self, self.image, self.color_param['yellow']['lower'], self.color_param['yellow']['upper'])
		self.point.xm, self.point.ym, self.areaM = self.color_segmentation(self, self.image, self.color_param['magenta']['lower'], self.color_param['magenta']['upper'])

		if self.areaY > 0 or self.areaM >0:
			
			if self.areaM > self.areaY:
				selected_color = 'magenta'
			elif self.areaM < self.areaY:
				selected_color = 'yellow'

			if self.selected_color == "magenta":
				# append magenta_machine to robot_machine
				self.robot_machine.update(self.magenta_machine)
				self.robot_state = "aproxima"

			elif self.selected_color == "yellow":
				# append yellow_machine to robot_machine
				self.robot_machine.update(self.yellow_machine)
				self.robot_state = "afasta"
		
		else:
			self.robot_state = 'rotate'

	def aproxima(self) -> None:
		"""
		Go to
		"""
		self.kp = 200
		self.w = self.image.shape[1]

		if self.laser_forward >= 0.2:
			
			erro = self.w/2 - self.point.xm
			self.twist.angular.z = float(erro) / self.kp 
			self.twist.linear.x = 0.2

			self.cmd_vel_pub.publish(self.vel)
		#else:
			#voltar a posicao original

	def afasta(self) -> None:
		"""
		Go away
		"""
		self.kp = 200
		self.w = self.image.shape[1]

		if self.laser_backwards >= 0.2:
			
			erro = self.w/2 - self.point.xy
			self.twist.angular.z = float(erro) / self.kp 
			self.twist.linear.x = -0.2

			self.cmd_vel_pub.publish(self.vel)
		#else:
			#voltar a posicao original
	
	def para(self) -> None:
		"""
		Stop the robot
		"""
		self.twist.angular.z = 0
		self.robot_state = 'checar'
		
	def center_on_coord(self):
		...

	def go_to_coord(self):
		...

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

		#self.temporodando = rospy.Time.now().to_sec()
		#rospy.Time.now().to_sec(rospy.Duration(10))
		#msg = String()
		#msg.data = f'{tempo} {self.count}'
		#rospy.loginfo(f'Ola, são {tempo} e estou publicando pela {self.contador} vez')
		#self.pub.publish(msg)

def main():
	rospy.init_node('Aleatorio')
	control = Control()
	rospy.sleep(1)

	while not rospy.is_shutdown():
		control.control()

if __name__=="__main__":
	main()

