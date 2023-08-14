#! /usr/bin/env python3
# -*- coding:utf-8 -*-

import rospy
from geometry_msgs.msg import Twist, Vector3, Point
from nav_msgs.msg import Odometry
from tf.transformations import euler_from_quaternion
import numpy as np

'''
roslaunch my_simulation pista23-1.launch
'''

v = 0.2  # Velocidade linear
w = 0.5  # Velocidade angular

class Control():
    def __init__(self):
        self.rate = rospy.Rate(250) # 250 Hz
        self.pointA = Point( x = 0, y = -1.9, z = 0) #coordenadas bifurcacao1
        self.pointB = Point( x = 0, y = 0.54, z = 0) #coordenadas bifurcacao2
        self.kp = 0.5 #ganho

        self.robot_state = 'center'
        self.state_machine = {
			'center': self.center,
			'goto': self.goto,
			'stop': self.stop
		}

        # Subscribers
        self.odom_sub = rospy.Subscriber("/odom",Odometry,self.odom_callback)        
        
        # Publisher
        self.cmd_vel_pub = rospy.Publisher('/cmd_vel',Twist,queue_size=1)


    def odom_callback(self, data: Odometry):
        self.odom = data
        self.x = data.pose.pose.position.x
        self.y = data.pose.pose.position.y
        self.z = data.pose.pose.position.z

        orientation_list = [data.pose.pose.orientation.x,
        					data.pose.pose.orientation.y,
        					data.pose.pose.orientation.z,
        					data.pose.pose.orientation.w]        
        self.roll, self.pitch, self.yaw = euler_from_quaternion(orientation_list)

        # convert yaw from [-pi, pi] to [0, 2pi]
        self.yaw = self.yaw % (2*np.pi)

    def get_angular_error(self):
		#calcular erro 
        x = self.point.x - self.x #ponto q desejamos - ponto em que estamos
        y = self.point.y - self.y
        theta = np.arctan2(y, x)     

        self.distance = np.sqrt(x**2 + y**2)
        self.err = np.rad2deg(theta - self.yaw)
        self.err = self.err % 360 #rest  

        self.twist.angular.z = self.err * self.kp
                
    def control(self) -> None:
        '''
		This function is called at least at {self.rate} Hz.
		'''
        self.twist = Twist()
        print(f'{self.robot_state}')
        self.state_machine[self.robot_state]()

        self.cmd_vel_pub.publish(self.twist)
        self.rate.sleep() # Sleeps the remaining time to keep the rate


if __name__ == "__main__":
    rospy.init_node("roda_exemplo")
    pub = rospy.Publisher("cmd_vel", Twist, queue_size=3)

    try:
        while not rospy.is_shutdown():
            vel = Twist(Vector3(v,0,0), Vector3(0,0,w))
            pub.publish(vel)
            rospy.sleep(2.0)
    except rospy.ROSInterruptException:
        print("Ocorreu uma exceção com o rospy")


    