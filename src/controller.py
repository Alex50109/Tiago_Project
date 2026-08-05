#!/usr/bin/env python

import rospy
from geometry_msgs.msg import Twist
import math
from sensor_msgs.msg import Image

def ticontrol():
    rospy.init_node('controller', anonymous=True)

    while rospy.Time.now() == rospy.Time(0) and not rospy.is_shutdown():
        rospy.loginfo_throttle(1.0, "Waiting for Gazebo simulation time to start...")
        rospy.sleep(0.1)

    cmd_pub = rospy.Publisher('/mobile_base_controller/cmd_vel', Twist, queue_size=10)
    ai_pub = rospy.Publisher('request_ai_assistance', Image, queue_size=10)      
    capture_surroundings(cmd_pub, ai_pub)
       

def capture_surroundings(cmd_pub:rospy.Publisher, ai_pub:rospy.Publisher):
    # Initialize the node    
    # Setup Publishers
    # TIAGo's camera topic (Verify this matches your specific simulation model)
    # Common alternatives: '/camera/rgb/image_raw' or '/rgbd_camera/color/image_raw'
    camera_topic = '/xtion/rgb/image_raw'
    
    # 45 degrees in radians
    target_angle_rad = math.pi / 4.0  
    
    # Angular velocity (radians per second)
    angular_speed = 0.5               
    
    # How long to spin to reach 45 degrees at the set speed
    rotation_time = target_angle_rad / angular_speed
    
    # Give the publishers a moment to establish connections to the ROS master
    rospy.sleep(1) 

    rospy.loginfo("Waiting for Gazebo camera plugin to come online...")
    try:
        rospy.wait_for_message(camera_topic, Image)
        rospy.loginfo("Camera is online!")
    except rospy.ROSInterruptException:
        return
        
    rospy.loginfo("Starting 360-degree scan (8 pictures at 45-degree intervals)")

    rate = rospy.Rate(10)
    
    for i in range(8):
        current_angle_deg = i * 45
        rospy.loginfo("Taking picture {}/8 at {} degrees...".format(i+1, current_angle_deg))
        
        try:
            # 1. Block and wait for a single image frame
            image_msg = rospy.wait_for_message(camera_topic, Image, timeout=5.0)
            
            # 2. Publish the single image to the AI topic
            ai_pub.publish(image_msg)
            rospy.loginfo("Picture {} published to 'request_ai_assistance'".format(i+1))
            
        except rospy.ROSException:
            rospy.logwarn("Timeout! Failed to get image from {}".format(camera_topic))
        
        # 3. Rotate 45 degrees (skip rotation on the final picture)
        
        rospy.loginfo("Rotating 45 degrees...")
        vel_msg = Twist()
        vel_msg.angular.z = angular_speed
        cmd_pub.publish(vel_msg)

        start_time = rospy.Time.now()
        while (rospy.Time.now() - start_time).to_sec() < rotation_time and not rospy.is_shutdown():
            cmd_pub.publish(vel_msg)
            rate.sleep()
            
            # Stop the robot securely by publishing an empty Twist (all zeros)
        cmd_pub.publish(Twist())
            
            # Pause for 1 second to let the camera physically stabilize before the next shot
        rospy.sleep(1.0)
            
    rospy.loginfo("AI Request Scan complete.")        

if __name__ == '__main__':
    try:
        ticontrol()
    except rospy.ROSInterruptException:
        pass