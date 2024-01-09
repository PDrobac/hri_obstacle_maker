#! /usr/bin/env python3
import numpy as np
import rospy
import moveit_commander
import math
from geometry_msgs.msg import PoseStamped, Point
from sensor_msgs.msg import Joy, PointCloud
from scipy.spatial.transform import Rotation
import matplotlib.pyplot as plt
from scipy.linalg import expm  # Import expm from scipy.linalg
from scipy.spatial.distance import cdist
from scipy.spatial.transform import Rotation
from scipy.optimize import minimize
from sklearn.linear_model import LinearRegression
import rosbag
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial import distance
import numpy as np

class Integrator:
    def create_scatter_plot(self, N = 100, inclination_coef=1, azimut_coef = 1, randomnes = 5, filterd = False, filter_threshold=5):
        #np.random.seed(2)
        # Create a 2D array with 100 elements and two columns of random float values between -10 and 10
        array_2d_float = np.random.uniform(-10, 10, size=(N, 2))

        ones_with_variance = np.random.normal(loc=0, scale=inclination_coef * randomnes+0.1, size=N)
        
        # Azimut and inclination rotation of points
        ones_with_variance += -azimut_coef * inclination_coef * array_2d_float[:, 0]
        ones_with_variance += azimut_coef * inclination_coef * array_2d_float[:, 1]

        # Vertically stack the arrays
        combined_array = np.vstack((array_2d_float.T, ones_with_variance)).T

        # Filter points based on coordinates
        if filterd == True:
            combined_array = combined_array[
                ~((combined_array[:, 0] > filter_threshold) & (combined_array[:, 1] > filter_threshold)) &
                ~((combined_array[:, 0] < -filter_threshold) & (combined_array[:, 1] < -filter_threshold)) &
                ~((combined_array[:, 0] < -filter_threshold) & (combined_array[:, 1] > filter_threshold)) &
                ~((combined_array[:, 0] > filter_threshold) & (combined_array[:, 1] < -filter_threshold))
            ]
        return combined_array
    
    def create_hexagon_scatter_plot(self, N=100, radius=5):
        # Generate random angles for points within the hexagon
        angles = np.linspace(0, 2 * np.pi, N)
        
        # Calculate X and Y coordinates based on the hexagon's radius
        x_coordinates = radius * np.cos(angles)
        y_coordinates = radius * np.sin(angles)

        # Add some randomness to the Z coordinate
        z_coordinates = np.random.normal(loc=0, scale=1, size=N)

        # Combine the coordinates into a 3D array
        combined_array = np.vstack((x_coordinates, y_coordinates, z_coordinates)).T

        return combined_array

 
    def process_trajectory_data(self, bag_file_path, desired_seq_numbers, distance_threshold=0.00000000001, time_interval=0.01):
        joy_topic = '/kalipen/joy'
        pose_topic = '/vrpn_client_node/Kalipen/pose'
        bag = rosbag.Bag(bag_file_path)

        desired_seq_numbers_timestamps = []

        # Store messages with specific sequence numbers
        print("Storing messages with specific sequence numbers")
        for topic, msg, timestamp in bag.read_messages(topics=[joy_topic]):
            # Check if the sequence number is in the desired list
            if hasattr(msg, 'header') and msg.header.seq in desired_seq_numbers:
                print(f"Time: {timestamp}, Sequence: {msg.header.seq}")
                desired_seq_numbers_timestamps.append(timestamp)

        # Store all messages from the desired topic within the specified time range and time interval
        filtered_positions = []
        for topic, msg, timestamp in bag.read_messages(topics=[pose_topic]):
            if desired_seq_numbers_timestamps and desired_seq_numbers_timestamps[0] < timestamp < desired_seq_numbers_timestamps[1]:
                x = msg.pose.position.x
                y = msg.pose.position.y
                z = msg.pose.position.z
                # Check if the point is far enough from all previously stored points
                min_distance = min(distance.euclidean([x, y, z], point[:3]) for point in filtered_positions) if filtered_positions else float('inf')
                # Convert Duration to float
                timestamp_float = timestamp.to_sec()
                # Check if the timestamp is within the desired time interval
                if min_distance > distance_threshold and (timestamp_float - filtered_positions[-1][3] >= time_interval) if filtered_positions else True:
                    filtered_positions.append([x, y, z, timestamp_float])
        bag.close()
        return np.array(filtered_positions)[:, :3]
    
    def fit_plane_and_get_normal(self, ax, data_uncentered, centroid):
        x = data_uncentered[:, :2]
        y = data_uncentered[:, 2]
        
        model = LinearRegression().fit(x, y)
        x_coefficient, y_coefficient = model.coef_
        z_coefficient = model.intercept_

        # Create a meshgrid for the plane
        x_plane = np.linspace(min(x[:, 1]), max(x[:, 1]), 100)
        y_plane = np.linspace(min(x[:, 0]), max(x[:, 0]), 100)
        x_plane, y_plane = np.meshgrid(x_plane, y_plane)
        z_plane = x_coefficient * x_plane + y_coefficient * y_plane + z_coefficient
        ax.plot_surface(x_plane, y_plane, z_plane, alpha=0.5, rstride=100, cstride=100, color='red', label='Fitted Plane')

        vectors = np.empty((0, 3))
        for x_y in [[1e30, 1e30], [1e30, -1e30]]:
            z_plane = x_coefficient * x_y[0] + y_coefficient * x_y[1] + z_coefficient
            vect = np.array([x_y[0], x_y[1], z_plane])
            vectors = np.vstack((vectors, vect))
            ax.quiver(centroid[0], centroid[1], centroid[2], vect[0], vect[1], vect[2], color='b', label='Perpendicular Vector')
        
        z_axis = np.cross(vectors[0], vectors[1])
        z_axis = z_axis / np.linalg.norm(z_axis)
        return z_axis
    
    def max_distances_from_plane(self, axis, centroid, data_uncenterd):
        max_distances = []
        for axis in axis:
            distances = []
            normalized_axis = axis/np.linalg.norm(axis)
            # Calculate all distances form points to a plane https://stackoverflow.com/a/55195238
            for i in range(len(data_uncenterd)):
                dist = np.linalg.norm(np.dot(data_uncenterd[i] -centroid, normalized_axis))
                distances.append(dist)
            # Find the maximum distance using np.argmax
            max_dist = distances[np.argmax(distances)]
            max_distances.append(max_dist)
        return max_distances
    
    def get_perpendicular_vectors(self, vector, ax, centroid, angle=1):
        # Find a perpendicular vector (using an arbitrary vector [1, 0, 0] for simplicity)
        perpendicular_vector = np.cross(vector, np.array([1, 0, 0]))

        # Normalize the perpendicular vector
        perpendicular_vector = perpendicular_vector / np.linalg.norm(perpendicular_vector)
        offset_perpendicular_vectors = []
        # Calculate a rotation matrix for a 360-degree rotation around the given vector
        for i in range(int(round(360/angle))):
            rotation_matrix = expm(np.cross(np.eye(3), vector / np.linalg.norm(vector) * np.radians(i*angle)))

            # Apply the rotation to the perpendicular vector
            offset_perpendicular_vector = rotation_matrix @ perpendicular_vector
            offset_perpendicular_vectors.append(offset_perpendicular_vector)

            # Plot the offset perpendicular vector
            ax.quiver(centroid[0], centroid[1], centroid[2], 0.1*offset_perpendicular_vector[0], 0.1*offset_perpendicular_vector[1], 0.1*offset_perpendicular_vector[2], color='g', label='Offset Perpendicular Vector')
        offset_perpendicular_vectors = np.vstack(offset_perpendicular_vectors)
        return offset_perpendicular_vectors

    def claculate_cube_volume(self, centroid, axis , data_uncenterd, ax):
        max_distances = self.max_distances_from_plane(axis, centroid, data_uncenterd)   
        cube_volume = np.prod(max_distances, axis=0)
        self.centroids_list.append(centroid)
        return cube_volume

    def run(self, cloud: PointCloud):
            data_uncenterd = np.array([[point.x, point.y, point.z] for point in cloud.points])
  
            fig = plt.figure(figsize=(12, 6))
            ax = fig.add_subplot(131, projection='3d')


            bag_file_path = '2023-12-19-10-37-38.bag'
            sequence = 41
            desired_seq_numbers = [sequence, sequence+1]
            distance_threshold=0.00000000001
            time_interval=0.01

            #data_uncenterd = self.process_trajectory_data(bag_file_path, desired_seq_numbers, distance_threshold=distance_threshold, time_interval=time_interval)
            
            #N = 100
            #inclination_coef=1#0.0001
            #data = self.create_scatter_plot(N = N, inclination_coef=inclination_coef, azimut_coef = 1, randomnes = 5, filterd = False, filter_threshold=5)
            #data = self.create_hexagon_scatter_plot(N=100, radius=5)
            #data = self.create_scatter_plot(N = 500, inclination_coef=0.01, azimut_coef = 1, randomnes = 1000, filter_threshold=5)
            #data_uncenterd = data + 10
            #data = data_uncenterd - centroid

            print(data_uncenterd)
            centroid = np.mean(data_uncenterd, axis=0)
            print("Centroid: ", centroid)

            z_axis = self.fit_plane_and_get_normal(ax, data_uncenterd, centroid)


            # Show the plot
            print("Normal Vector:", z_axis)
            ax.quiver(centroid[0], centroid[1], centroid[2], 2*z_axis[0], 2*z_axis[1], 2*z_axis[2], color='b', label='Perpendicular Vector') 
            
            angle = 1

            z_axis_perpendicular_vectors = self.get_perpendicular_vectors(z_axis, ax, centroid, angle=angle)
            max_distances_all = []
            for y_axis in z_axis_perpendicular_vectors:
                x_axis = np.cross(y_axis,z_axis)
                max_distances = self.max_distances_from_plane([z_axis, x_axis, y_axis], centroid, data_uncenterd)  
                max_distances_all.append(max_distances)
            max_distances_all = np.vstack(max_distances_all)

            cube_volume = np.prod(max_distances_all, axis=1)
            best_cube_indicies = np.argmin(cube_volume)
            max_distances = max_distances_all[best_cube_indicies]

            y_axis = z_axis_perpendicular_vectors[best_cube_indicies]
            ax.quiver(centroid[0], centroid[1], centroid[2], 2*y_axis[0], 2*y_axis[1], 2*y_axis[2], color='y', label='Perpendicular Vector')
            x_axis = np.cross(y_axis,z_axis)
            ax.quiver(centroid[0], centroid[1], centroid[2], 2 *x_axis[0], 2*x_axis[1], 2*x_axis[2], color='g', label='Perpendicular Vector')
            
            ax_max_distances = fig.add_subplot(132)
            #ax_max_distances.plot(max_distances_z_direction, label='Max Distances Z')
            ax_max_distances.plot(cube_volume, label='Cube Volume')
            ax_max_distances.plot(max_distances_all[:, 2], label='Max Distances X')
            ax_max_distances.plot(max_distances_all[:, 1], label='Max Distances Y')
            ax_max_distances.plot(max_distances_all[:, 0], label='Max Distances Z')

            ax_max_distances.set_xlabel('Angle')
            ax_max_distances.legend()



            self.centroids_list = []
            result = minimize(lambda x: self.claculate_cube_volume(x, [z_axis, x_axis, y_axis], data_uncenterd, ax), centroid, method='Nelder-Mead', tol=1e-5)
            centroids_array = np.array(self.centroids_list)
            ax_centroids = fig.add_subplot(133)
            ax_centroids.plot(centroids_array[:, 0], label='X')
            ax_centroids.plot(centroids_array[:, 1], label='Y')
            ax_centroids.plot(centroids_array[:, 2], label='Z')
            ax_centroids.legend()
            print("Optimized centroid was moved:", result.x-centroid)
            print("Optimal solution (new centroid):", result.x)
            print("Optimal objective value:", result.fun)
            centroid = result.x

            # max distances from x, y and z planes
            print("Max distance in X direction:", max_distances[1])
            print("Max distance in Y direction:", max_distances[2])
            print("Max distance in Z direction:", max_distances[0])

            max_distances = np.array(self.max_distances_from_plane([z_axis, x_axis, y_axis], centroid, data_uncenterd)) 

            # max distances from x, y and z planes
            print("Max distance in X direction after optimization:", max_distances[1])
            print("Max distance in Y direction after optimization:", max_distances[2])
            print("Max distance in Z direction after optimization:", max_distances[0])

            # Scatter plot with variance
            ax.scatter(data_uncenterd[:, 0], data_uncenterd[:, 1], data_uncenterd[:, 2], c=data_uncenterd[:, 2], cmap='viridis')

            limit=30
            ax.set_xlim([-limit, limit])
            ax.set_ylim([-limit, limit])
            ax.set_zlim([-limit, limit])

            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')

            # Construct the rotation matrix
            R = np.vstack((x_axis, y_axis ,z_axis)).T
            print("Rotation matrix:\n", R)

            max_distances *= 2 # distance from centroid is half the size of the box sides
            dim1 = max_distances[1] 
            dim2 = max_distances[2]
            dim3 = max_distances[0]

            box_pose = PoseStamped()
            box_pose.header.frame_id = "world"
            box_pose.pose.position.x = centroid[0] 
            box_pose.pose.position.y = centroid[1]
            box_pose.pose.position.z = centroid[2] 

            qx, qy, qz, qw = Rotation.from_matrix(R).as_quat()
            box_pose.pose.orientation.x = qx
            box_pose.pose.orientation.y = qy
            box_pose.pose.orientation.z = qz
            box_pose.pose.orientation.w = qw
            

            box_name = "box"
            self.scene.add_box(box_name, box_pose, size=(dim1, dim2, dim3))
            
            box_pose_0 = PoseStamped()
            box_pose_0.header.frame_id = "world"
            box_pose_0.pose.position.x = x_axis[0]*1+centroid[0]
            box_pose_0.pose.position.y = x_axis[1]*1+centroid[1]
            box_pose_0.pose.position.z = x_axis[2]*1+centroid[2]


            box_pose_1 = PoseStamped()
            box_pose_1.header.frame_id = "world"
            box_pose_1.pose.position.x = y_axis[0]*1+centroid[0]
            box_pose_1.pose.position.y = y_axis[1]*1+centroid[1]
            box_pose_1.pose.position.z = y_axis[2]*1+centroid[2]

            box_pose_2 = PoseStamped()
            box_pose_2.header.frame_id = "world"
            box_pose_2.pose.position.x = z_axis[0]*1+centroid[0]
            box_pose_2.pose.position.y = z_axis[1]*1+centroid[1]
            box_pose_2.pose.position.z = z_axis[2]*1+centroid[2]

            box_pose_3 = PoseStamped()
            box_pose_3.header.frame_id = "world"
            box_pose_3.pose.position.x = centroid[0]
            box_pose_3.pose.position.y = centroid[1]
            box_pose_3.pose.position.z = centroid[2]

            self.scene.add_box(box_name, box_pose, size=(dim1, dim2, dim3))
        
            for size in [(0.015, 0.015, 0.1), (0.015, 0.1, 0.015), (0.1, 0.015, 0.015)]:
                for i, point in enumerate(data_uncenterd):
                    box_pose = PoseStamped()
                    box_pose.header.frame_id = "world"
                    box_pose.pose.position.x = point[0]
                    box_pose.pose.position.y = point[1]
                    box_pose.pose.position.z = point[2]
                    
                    box_name = f"box{i,size}"
                    self.scene.add_box(box_name, box_pose, size=size) 

            self.scene.add_box("box0", box_pose_0, size=(0.1, 0.1, 0.1))
            self.scene.add_box("box1", box_pose_1, size=(0.1, 0.1, 0.1))
            self.scene.add_box("box2", box_pose_2, size=(0.1, 0.1, 0.1))
            self.scene.add_box("box3", box_pose_3, size=(0.1, 0.1, 0.1))

            #plt.show()
            
            #rospy.signal_shutdown("Script completed")  

            #pass

    def __init__(self):
        rospy.loginfo("Integrator started")
        rospy.Subscriber('/obstacle', PointCloud, self.run, queue_size=1)

        self.clicked = 1 #0
        self.counter = 0
        self.points = list()
        self.pose = PoseStamped()
        self.scene = moveit_commander.PlanningSceneInterface()

if __name__ == '__main__':
    try:
        rospy.init_node('integrator_node')
        integrator = Integrator()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
