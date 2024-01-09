#! /usr/bin/env python
import rospy
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import Joy, PointCloud

class Controller:     
    def pose_callback(self, pose: PoseStamped):
        #rospy.loginfo(pose.pose)
        if(self.collect):
            self.points.points.append(pose.pose.position)

    def click_callback(self, click: Joy):
        #rospy.loginfo("click_callback")
        if (click.buttons[0] == 1):
            self.collect = True
        elif (click.buttons[0] == 0):
            self.collect = False
            self.pub.publish(self.points)
            self.rate.sleep()
            self.points = PointCloud()

    def __init__(self):
        rospy.loginfo("Controller started")
        rospy.Subscriber('/vrpn_client_node/Kalipen/pose', PoseStamped, self.pose_callback, queue_size=1)
        rospy.Subscriber('/kalipen/joy', Joy, self.click_callback, queue_size=1)
        self.pub = rospy.Publisher('/obstacle', PointCloud, queue_size=10)

        self.rate = rospy.Rate(100)

        self.collect = False
        self.points = PointCloud()

        self.pub.publish

if __name__ == '__main__':
    try:
        rospy.init_node('controller')
        controller = Controller()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
