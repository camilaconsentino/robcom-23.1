#! /usr/bin/env python3
# -*- coding:utf-8 -*-

import rospy
from geometry_msgs.msg import Twist, Vector3
from std_msgs.msg import Float64

if __name__ == "__main__":
    #inicializando o node com o ROS
    rospy.init_node("garra")
    #Criando o publisher para controle do braço
    braco = rospy.Publisher("/joint1_position_controller/command", Float64, queue_size=1)
    
    #criando o publisher para controle da pinça
    pinca = rospy.Publisher("/joint2_position_controller/command", Float64, queue_size=1)

    #loop para movimentar a garra do robo
    try:
        while not rospy.is_shutdown():
            braco.publish(1.5) ## para baixo
            rospy.sleep(0.5)
            braco.publish(-1.5) ## para cima
            rospy.sleep(0.5)
            braco.publish(0.0) ## para frente
            rospy.sleep(0.5)
            pinca.publish(-1.0) #aberta?
            rospy.sleep(0.5)
            pinca.publish(0.0) #fechada
            rospy.sleep(0.5)


    except rospy.ROSInterruptException:
        print("Ocorreu uma exceção com o rospy")