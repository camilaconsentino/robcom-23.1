import cv2
import numpy as np
import matplotlib.pyplot as plt

def color_segmentation(bgr):
		""" 
		Use HSV color space to segment the image and find the center of the object.

		Args:
			bgr (np.ndarray): image in BGR format
		"""
		
		lower_hsv = np.array([225//2,50,50],dtype=np.uint8) 		
		upper_hsv = np.array([255//2,255,255],dtype=np.uint8)

		kernel = np.ones((5,5),np.uint8)

		hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)

		mask = cv2.inRange(hsv, lower_hsv, upper_hsv)
		mask = cv2.morphologyEx(mask,cv2.MORPH_OPEN, kernel)
		mask = cv2.morphologyEx(mask,cv2.MORPH_CLOSE, kernel)

		# find contours
		contours,_ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

		return mask, contours

		'''
		if len(contours) > 0:
			# find contour with max area
			cnt = max(contours, key = lambda x: cv2.contourArea(x))

			# Find the center
			M = cv2.moments(cnt)
			x = int(M['m10']/M['m00'])
			y = int(M['m01']/M['m00'])

			print("tem contorno")
		else:
			x = -1
			y = -1
		'''

img = cv2.imread("/home/borg/entregas-robotica/APS04/imagem_debug1.png")
			
mask, cts = color_segmentation(img)

print(cts)
plt.imshow(mask)
plt.show()