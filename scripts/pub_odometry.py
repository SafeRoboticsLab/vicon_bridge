#!/usr/bin/env python
import rospy
from geometry_msgs.msg import TransformStamped
from nav_msgs.msg import Odometry
import numpy as np
import scipy
from scipy.spatial.transform import Rotation as R

def get_T(position, orientation):
    quat = [orientation.x, orientation.y, orientation.z, orientation.w]
    trans = [position.x, position.y, position.z]
    T = np.eye(4)
    T[:3, :3] = R.from_quat(quat).as_matrix()
    T[:3, 3] = trans
    return T

def SE3_log(T):
    # Give T is SE(3), return the  t = (w, v) in se(3)
    t_hat = scipy.linalg.logm(T) 
    w = np.array([t_hat[2, 1], t_hat[0, 2], t_hat[1, 0]])
    v = t_hat[:3, 3]
    return w, v
    

class PublishOdometry:
    def __init__(self):
        self.vicon_topic = rospy.get_param("~vicon_topic", "/vicon/zed/zed")
        self.pub_topic = rospy.get_param("~pub_topic", "/vicon/pose")
        self.freq = rospy.get_param("~pub_freq", 50)
        
        self.subscriber = rospy.Subscriber(self.vicon_topic, TransformStamped, self.callback)
        self.publisher = rospy.Publisher(self.pub_topic, Odometry, queue_size=10)
        
        self.t_prev = None
        self.t_last_pub = None
        self.pose_prev = None
        
    def callback(self, vicon_pose):
        
        # convert to vicon pose to odometry
        t_cur = vicon_pose.header.stamp.to_sec()
        Pose_cur = get_T(vicon_pose.transform.translation, vicon_pose.transform.rotation)
        
        if self.t_last_pub is not None:
            dt = t_cur - self.t_prev
            
            # For discrete time, we have
            # T_1 = T_0 * exp(dt * twist^)
            # https://natanaso.github.io/ece276a2020/ref/ECE276A_12_SO3_SE3.pdf
            dPose = np.dot(np.linalg.inv(self.pose_prev), Pose_cur)
            dw, dv = SE3_log(dPose)
            
            w = dw / dt
            v = dv / dt
            
            dt_pub = t_cur - self.t_last_pub
            if dt_pub >= 1.0 / self.freq:
                odom = Odometry()
                odom.header.stamp = vicon_pose.header.stamp
                
                odom.pose.pose.position = vicon_pose.transform.translation
                odom.pose.pose.orientation = vicon_pose.transform.rotation
                
                odom.twist.twist.linear.x = v[0]
                odom.twist.twist.linear.y = v[1]
                odom.twist.twist.linear.z = v[2]
                odom.twist.twist.angular.x = w[0]
                odom.twist.twist.angular.y = w[1]
                odom.twist.twist.angular.z = w[2]
                
                self.publisher.publish(odom)
                self.t_last_pub = t_cur
        else:
            self.t_last_pub = t_cur
            
        self.t_prev= t_cur
        self.pose_prev = Pose_cur
            
        
def main():
    rospy.init_node('ViconOdometry')
    rospy.loginfo("Start Vicon Odometry node")


    PublishOdometry()
    rospy.spin()


if __name__ == '__main__':
    main()    
