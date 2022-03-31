#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

from numpy import average
import numpy as np
import rospy
from lidar_detect.msg import lidar_3dbox_array
from std_msgs.msg import Bool


class lidar_obj_Subscriber(object):
    def __init__(self) -> None:
        self.sub = rospy.Subscriber("/lidarobj",
                                     lidar_3dbox_array,
                                     self.callback, queue_size=5)
        self.pub = rospy.Publisher('/lidar_car_stop', Bool, queue_size=10)
        self.scoresqueen = []
    def callback(self, msg):
        assert isinstance(msg, lidar_3dbox_array)
        stop = False
        score = []
        for box in msg.boxes:
            score.append(box.score)
        if len(score)==0:
            self.add_score(0)
            self.pub.publish(stop)
            return
        self.add_score(max(score))
        score_max_index = score.index(max(score))
        # print(np.var(self.scoresqueen))
        
        # if len(self.scoresqueen)==10 and average(self.scoresqueen)>0.3 and np.var(self.scoresqueen)<0.04:
        if len(self.scoresqueen)==10 and average(self.scoresqueen)>0.3:
            position =  msg.boxes[score_max_index].position
            x = position.split(',')[0]
            y = position.split(',')[1]
            # print(position)
            print('找到障碍物')
            if 0<float(x)<=1 and abs(float(y))<0.35:
                print('停车')
                stop = True
            else:
                print('障碍物不在范围内')
        self.pub.publish(stop)
    def add_score(self, score):
        if len(self.scoresqueen)<10:
            self.scoresqueen.append(score)
        else:
            self.scoresqueen.pop(0)
            self.scoresqueen.append(score)


if __name__ =='__main__':
    print('strat')
    rospy.init_node("lidar_car_stop")
    lidar_obj_Subscriber()
    rospy.spin()
