#! /usr/bin/env python
import numpy as np
import rospy
import moveit_commander
import math
from geometry_msgs.msg import PoseStamped, Point
from sensor_msgs.msg import Joy, PointCloud
from scipy.spatial.transform import Rotation

class Integrator:
    def euler_to_quaternion(self, yaw, pitch, roll):
        yaw *= 0.01745329
        pitch *= 0.01745329
        roll *= 0.01745329

        qx = np.sin(roll/2) * np.cos(pitch/2) * np.cos(yaw/2) - np.cos(roll/2) * np.sin(pitch/2) * np.sin(yaw/2)
        qy = np.cos(roll/2) * np.sin(pitch/2) * np.cos(yaw/2) + np.sin(roll/2) * np.cos(pitch/2) * np.sin(yaw/2)
        qz = np.cos(roll/2) * np.cos(pitch/2) * np.sin(yaw/2) - np.sin(roll/2) * np.sin(pitch/2) * np.cos(yaw/2)
        qw = np.cos(roll/2) * np.cos(pitch/2) * np.cos(yaw/2) + np.sin(roll/2) * np.sin(pitch/2) * np.sin(yaw/2)

        return [qx, qy, qz, -qw]
     
    def pose_callback(self, pose: PoseStamped):
        #print(pose.pose)
        self.pose = pose

    def click_callback(self, click: Joy):
        print("click_callback")
        if (click.buttons[0] == 1):
            self.clicked = True

    def cube(self):
        x0, y0, z0 = p0 = np.asarray(self.points[0])
        x1, y1, z1 = p1 = np.asarray(self.points[1])
        p2 = np.asarray(self.points[2])
        p3 = np.asarray(self.points[3])

        v1 = p1 - p0

        x2, y2, z2 = p2_f = np.cross(v1/np.linalg.norm(v1), -np.cross(v1/np.linalg.norm(v1), p2-p0))
        v2 = p2_f - p0

        n = np.cross(v1, v2)
        n = n/np.linalg.norm(n)
        x3, y3, z3 = p3_f = np.dot(p3-p0, n)*n
        v3 = p3_f - p0

        dim1 = math.dist(p0, p1)
        dim2 = math.dist(p0, p2_f)
        dim3 = math.dist(p0, p3_f)

        box_pose = PoseStamped()
        box_pose.header.frame_id = "world"

        box_pose.pose.position.x = (x0+x1+x2+x3)/2
        box_pose.pose.position.y = (y0+y1+y2+y3)/2
        box_pose.pose.position.z = (z0+z1+z2+z3)/2

        if (np.dot(n, v3) < 0):
            v3 *= -1

        v1_n = v1/np.linalg.norm(v1)
        v2_n = v2/np.linalg.norm(v2)
        v3_n = v3/np.linalg.norm(v3)

        M = np.append([v1_n, v2_n], [v3_n], axis=0)

        r =  Rotation.from_matrix(M)
        x, y, z = r.as_euler("xyz",degrees=True)

        qx, qy, qz, qw = integrator.euler_to_quaternion(z, y, x)

        box_pose.pose.orientation.x = qx
        box_pose.pose.orientation.y = qy
        box_pose.pose.orientation.z = qz
        box_pose.pose.orientation.w = qw

        box_name = "box_" + str(self.object_ctr)
        self.object_ctr += 1

        self.scene.add_box(box_name, box_pose, size=(dim1, dim2, dim3))

    def plane(self):
        x0, y0, z0 = p0 = np.asarray(self.points[0])
        x1, y1, z1 = p1 = np.asarray(self.points[1])
        p2 = np.asarray(self.points[2])

        v1 = p1 - p0

        x2, y2, z2 = p2_f = np.cross(v1/np.linalg.norm(v1), -np.cross(v1/np.linalg.norm(v1), p2-p0))
        v2 = p2_f - p0

        n = np.cross(v1, v2)
        n = n/np.linalg.norm(n)

        dim1 = math.dist(p0, p1)
        dim2 = math.dist(p0, p2_f)
        dim3 = 0.0001

        box_pose = PoseStamped()
        box_pose.header.frame_id = "world"

        box_pose.pose.position.x = (x0+x1+x2)/2
        box_pose.pose.position.y = (y0+y1+y2)/2
        box_pose.pose.position.z = (z0+z1+z2)/2

        v1_n = v1/np.linalg.norm(v1)
        v2_n = v2/np.linalg.norm(v2)

        M = np.append([v1_n, v2_n], [n], axis=0)

        r =  Rotation.from_matrix(M)
        x, y, z = r.as_euler("xyz",degrees=True)

        qx, qy, qz, qw = integrator.euler_to_quaternion(z, y, x)

        box_pose.pose.orientation.x = qx
        box_pose.pose.orientation.y = qy
        box_pose.pose.orientation.z = qz
        box_pose.pose.orientation.w = qw

        box_name = "plane_" + str(self.object_ctr)
        self.object_ctr += 1

        self.scene.add_box(box_name, box_pose, size=(dim1, dim2, dim3))

    def collect_points(self, num: int):
        while (self.point_ctr < num):
            if(self.clicked):
                self.clicked = False
                curr_pose = Point()
                curr_pose = self.pose.pose
                self.points.append(curr_pose)
                self.point_ctr += 1
        self.point_ctr = 0

    def define_points(self, object: str):
        self.points.append((2,1,2))
        self.points.append((3,3,0))
        self.points.append((4,1,1))
        if (object == "cube"):
            self.points.append((-1,0,0))

    def plane_callback(self, points: PointCloud):
        self.points = np.asarray(points.points)
        integrator.plane()

    def box_callback(self, points: PointCloud):
        self.points = np.asarray(points.points)
        integrator.box()

    def run(self):
        while not rospy.is_shutdown():
            integrator.define_points("cube")
            integrator.cube()
        #integrator.plane()

            pass

    def __init__(self):
        rospy.loginfo("Integrator started")
        rospy.Subscriber('/vrpn_client_node/Kalipen/pose', PoseStamped, self.pose_callback, queue_size=1)
        rospy.Subscriber('/kalipen/joy', Joy, self.click_callback, queue_size=1)
        rospy.Subscriber('/objector/plane', PointCloud, self.plane_callback, queue_size=1)
        rospy.Subscriber('/objector/box', PointCloud, self.box_callback, queue_size=1)

        self.clicked = False
        self.point_ctr = 0
        self.object_ctr = 0
        self.points = list()
        self.pose = PoseStamped()
        self.scene = moveit_commander.PlanningSceneInterface()

if __name__ == '__main__':
    try:
        rospy.init_node('integrator_node')
        integrator = Integrator()
        integrator.run()
    except rospy.ROSInterruptException:
        pass
