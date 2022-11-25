#!/usr/bin/env python
import rospy
import message_filters
from geometry_msgs.msg import PoseStamped, TransformStamped
from IPython import display
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.transform import Rotation as R

def get_T(position, orientation):
    quat = [orientation.x, orientation.y, orientation.z, orientation.w]
    trans = [position.x, position.y, position.z]
    T = np.eye(4)
    T[:3, :3] = R.from_quat(quat).as_matrix()
    T[:3, 3] = trans
    return T

class PoseNode:
    def __init__(self):
        self.t = []
        self.x_slam = []
        self.y_slam = []
        self.z_slam = []
        
        self.x_vicon = []
        self.y_vicon = []
        self.z_vicon = []
        
        # initial time and pose
        self.t0 = None
        self.vicon2slam = np.array([[1,0,0,0],
                                    [0,1,0,0],
                                    [0,0,1,0],
                                    [0,0,0,1]])
        
        self.T_vicon0 = None
        self.T_slam0 = None
        
        # setup plot        
        plt.ion()
        self.fig, (self.a_x, self.a_y, self.a_z) = plt.subplots(3, 1)
        
        self.start_listening()
    
    def start_listening(self):
        rospy.init_node("SLAM_pose_listener", anonymous=True)
        slam_sub = message_filters.Subscriber("/slam_pose", PoseStamped)
        vicon_sub = message_filters.Subscriber("/vicon/zed/zed", TransformStamped)
        
        ts = message_filters.ApproximateTimeSynchronizer([slam_sub, vicon_sub], 10, 0.05, allow_headerless=True)
        ts.registerCallback(self.callback)
    
    def callback(self, slam_pose, vicon_pose):
        if self.t0 is None:
            self.t0 = slam_pose.header.stamp.to_sec()
            self.T_vicon0 = np.linalg.inv(get_T(vicon_pose.transform.translation, 
                                                vicon_pose.transform.rotation)*self.vicon2slam)
            self.T_slam0 = np.linalg.inv(get_T(slam_pose.pose.position, slam_pose.pose.orientation))
            
        self.t.append(slam_pose.header.stamp.to_sec() - self.t0)
        T_slam = self.T_slam0*get_T(slam_pose.pose.position, slam_pose.pose.orientation)
        T_vicon = self.T_vicon0*(get_T(vicon_pose.transform.translation,
                                       vicon_pose.transform.rotation)*self.vicon2slam)
        
        self.x_slam.append(T_slam[0,3])
        self.y_slam.append(T_slam[1,3])
        self.z_slam.append(T_slam[2,3])
        
        self.x_vicon.append(T_vicon[0,3])
        self.y_vicon.append(T_vicon[1,3])
        self.z_slam.append(T_vicon[2,3])
        
        self.show_plot()
    
    def show_plot(self):
        display.clear_output(wait = True)
        display.display(self.fig)
        plt.clf()
        
        self.a_x.plot(self.t, self.x_slam, 'r-', label="slam")
        self.a_x.plot(self.t, self.x_vicon, 'b--', label="vicon")
        
        self.a_y.plot(self.t, self.y_slam, 'r-', label="slam")
        self.a_y.plot(self.t, self.y_vicon, 'b--', label="vicon")
        
        
        self.a_z.plot(self.t, self.z_slam, 'r-', label="slam")
        self.a_z.plot(self.t, self.z_vicon, 'b--', label="vicon")
            
        plt.pause(0.001)
        



    
if __name__ == "__main__":
    listener = PoseNode()
    rospy.spin()
   