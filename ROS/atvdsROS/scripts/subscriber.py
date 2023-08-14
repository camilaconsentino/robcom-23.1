#!/usr/bin/env python3

import rospy
from std_msgs.msg import String

""" 
Running each line in a different terminal
	roslaunch roscore
	rosrun modulo4 publisher.py
	rosrun modulo4 subscriber.py
"""

class Subscriber():
	def __init__(self):
		self.rate = rospy.Rate(250) # 250 Hz

		# Subscribers
		self.sub = rospy.Subscriber("publisher", String, self.callback) #se inscrevendo no topico publisher criado

		# Publishers
	
	def callback(self, msg)	-> None:
		self.time, self.contador = msg.data.split(' ')

	def control(self) -> None:
		'''
		This function is called at least at {self.rate} Hz.
		'''
		time = rospy.Time.now().to_sec() #tempo atual
		delta_time = time - float(self.time) #tempo atual - tempo no momento em que o n'o se increveu
		rospy.loginfo(f'Ola, estou recebendo a mensagem: {self.contador} e se passaram {delta_time} segundos')
		
		self.rate.sleep()

def main():
	rospy.init_node('subscribers')
	control = Subscriber()
	rospy.sleep(1)

	while not rospy.is_shutdown():
		control.control()

if __name__=="__main__":
	main()

