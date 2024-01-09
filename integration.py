#! /usr/bin/env python
import rospy
import moveit_commander
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import Joy 

class Integrator:
     
    def pose_callback(self, pose: PoseStamped):
        #print(pose.pose)
        self.pose = pose

    def click_callback(self, click: Joy):
        print("click_callback")
        if (click.buttons[0] == 1):
            self.clicked = 1

    def run(self):
        while not rospy.is_shutdown():

            #print(self.clicked)
            if (self.clicked == 1):
                self.clicked = 0
                #print(self.pose.pose)
                box_pose = PoseStamped()
                box_pose.header.frame_id = "world"
                box_pose.pose = self.pose.pose
                box_name = "box"
                self.scene.add_box(box_name, box_pose, size=(0.2, 0.2, 0.2))
            pass

    def __init__(self):
        rospy.loginfo("Integrator started")
        rospy.Subscriber('/vrpn_client_node/Kalipen/pose', PoseStamped, self.pose_callback, queue_size=1)
        rospy.Subscriber('/kalipen/joy', Joy, self.click_callback, queue_size=1)

        self.clicked = 0
        self.pose = PoseStamped()
        self.scene = moveit_commander.PlanningSceneInterface()

if __name__ == '__main__':
    try:
        rospy.init_node('integrator_node')
        integrator = Integrator()
        integrator.run()
    except rospy.ROSInterruptException:
        pass
