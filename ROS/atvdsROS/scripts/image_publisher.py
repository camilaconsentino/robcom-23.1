#! /usr/bin/env python3
# -*- coding:utf-8 -*-

import rospy

import numpy as np
import cv2
from geometry_msgs.msg import ???
from sensor_msgs.msg import ???
from cv_bridge import CvBridge, CvBridgeError
import numpy as np

""" 
Running
	roslaunch my_simulation pista_s2.launch
	rosrun modulo4 cor.py
"""

class ImagePublisher():
	def __init__(self):
		self.rate = rospy.Rate(250) # 250 Hz

		# HSV Filter
		self.lower_hsv = np.array([100,60,60],dtype=np.uint8) # Blue
		self.upper_hsv = np.array([140,255,255],dtype=np.uint8)
		self.kernel = np.ones((5,5),np.uint8)

		# Image
		self.point = ???

		# Subscribers
		self.bridge = CvBridge()
		self.image_sub = rospy.Subscriber('/camera/image/compressed', ???, self.image_callback, queue_size=1, buff_size = 2**24)
		
		# Publishers
		self.image_pub = rospy.Publisher(???, ???, queue_size=1)
		self.point_pub = rospy.Publisher(???, ???, queue_size=1)
		
	def image_callback(self, msg: ???) -> None:
		"""
		Callback function for the image topic
		"""
		try:
			cv_image = self.bridge.compressed_imgmsg_to_cv2(msg, "bgr8")
		except CvBridgeError as e:
			print(e)
		
		self.color_segmentation(cv_image)

		self.image_pub.publish(self.bridge.cv2_to_compressed_imgmsg(cv_image))



	def color_segmentation(self,bgr: np.ndarray) -> None:
		""" 
		Use HSV color space to segment the image and find the center of the object.

		Args:
			bgr (np.ndarray): image in BGR format
		
		Returns:
			np.ndarray: mask of the creeper
		"""
		self.point = Point() # Reseta o ponto #???
		hsv = cv2.ctvColor(bgr, cv2.BGR2HSV) #???
		mask = cv2.inRange(hsv, self.lower_hsv, self.upper_hsv)
		mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self.kernel)
		mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self.kernel)
		mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self.kernel)
		cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAI)

		if len(cnts) > 0:
			cnt = max(cnts, key=lambda x: cv2.contourArea(x))

			M = cv2.moments(cnt)
			self.point.x = int(M['m10'] / M['m00']) 
			self.point.y = int(M['m01'] / M['m00'])

		else:
			self.point.x = 

	def control(self) -> None:
		'''
		This function is called at least at {self.rate} Hz.
		'''
		
		self.point_pub.publish(self.point)
		???
		
		self.rate.sleep()

def main():
	rospy.init_node('Controler')
	control = ImagePublisher()
	rospy.sleep(1)

	while not rospy.is_shutdown():
		control.control()

if __name__=="__main__":
	main()

