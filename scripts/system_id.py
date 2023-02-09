#!/usr/bin/env python
import rospy
from geometry_msgs.msg import TransformStamped
from nav_msgs.msg import Odometry
import numpy as np
import scipy
from scipy.spatial.transform import Rotation as R
import message_filters
from racecar_msgs.msg import ServoMsg 
from vicon_bridge.msg import SystemMsg
from datetime import datetime 
import csv
import os


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
    

class SystemID:
    def __init__(self):
        self.vicon_topic = rospy.get_param("~vicon_topic", "/vicon/NX14/NX14")
        self.control_topic = rospy.get_param("~control_topic", "/servo/g29_control")
        self.pub_topic = rospy.get_param("~pub_topic", "/vicon/system_id")
        
        self.max_v = rospy.get_param("~max_v", 5.0)
        self.min_v = rospy.get_param("~min_v", -0.1)
        self.max_accel = rospy.get_param("~max_accel", 10.0)
        
        vicon_sub = message_filters.Subscriber(self.vicon_topic, TransformStamped)
        control_sub = message_filters.Subscriber(self.control_topic, ServoMsg)
        self.publisher = rospy.Publisher(self.pub_topic, SystemMsg, queue_size=10)
        
        ts = message_filters.ApproximateTimeSynchronizer([vicon_sub, control_sub], 10, 0.1, allow_headerless=True)
        ts.registerCallback(self.callback)
        
        file_name = os.path.join(os.path.dirname(__file__), 'system_id'+datetime.now().strftime("%Y%m%d-%H%M%S")+'.csv')
        rospy.loginfo("Saving data to %s", file_name)
        self.csv_file = open(file_name, 'w')

        self.csv_writer = csv.writer(self.csv_file)
        self.csv_writer.writerow(['throttle', 'steer', 'accel', 'v', 'w'])

        self.t_prev = None
        self.v_prev = None
        self.pose_prev = None
        
    def callback(self, vicon_pose, control_msg):
        throttle = control_msg.throttle
        steer = control_msg.steer
        
        # convert to vicon pose to odometry
        t_cur = vicon_pose.header.stamp.to_sec()
        Pose_cur = get_T(vicon_pose.transform.translation, vicon_pose.transform.rotation)
        
        if self.t_prev is not None:
            dt = t_cur - self.t_prev
            
            # For discrete time, we have
            # T_1 = T_0 * exp(dt * twist^)
            # https://natanaso.github.io/ece276a2020/ref/ECE276A_12_SO3_SE3.pdf
            dPose = np.dot(np.linalg.inv(self.pose_prev), Pose_cur)
            dw, dv = SE3_log(dPose)
            
            w = dw[-1] / dt
            v = dv[0] / dt
            
            if self.v_prev is not None:

                accel = np.round((v - self.v_prev),5) / dt

                # check if the data is valid
                valid = v<=self.max_v and v>=self.min_v and abs(accel)<self.max_accel and abs(throttle)< 1 and abs(steer)<1
                
                if valid:
                    msg = SystemMsg()
                    msg.header.stamp = rospy.Time.now()
                    msg.throttle = throttle
                    msg.steer = steer
                    msg.accel = accel
                    msg.vel = v
                    msg.w = w
                    self.publisher.publish(msg)
                    self.csv_writer.writerow([throttle, steer, accel, v, w])
                    self.csv_file.flush()
                    
            self.v_prev = v
        self.t_prev= t_cur
        self.pose_prev = Pose_cur
        
    def run(self):
        while not rospy.is_shutdown():
            rospy.sleep(0.01)
        self.csv_file.close()
        print("Closed csv")
            
        
def main():
    rospy.init_node('SystemID')
    rospy.loginfo("Start SystemID node")


    systemid = SystemID()
    systemid.run()
    


if __name__ == '__main__':
    main()    
