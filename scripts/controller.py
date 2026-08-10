#!/usr/bin/env python

import rospy
import math
import actionlib
from geometry_msgs.msg import Twist
from sensor_msgs.msg import Image
from tiago_project.msg import ControllerSpinAction, ControllerSpinFeedback
from tiago_project.msg import ControllerNavigateAction, ControllerNavigateFeedback

class Controller:
    def __init__(self):
        self.busy = False

        self.cmd_pub = rospy.Publisher('/mobile_base_controller/cmd_vel', Twist, queue_size=10)

        self.spin_server = actionlib.SimpleActionServer(
            'spin',
            ControllerSpinAction,
            execute_cb=self.spin_callback,
            auto_start=False
        )
        self.spin_server.start()

        self.navigate_server = actionlib.SimpleActionServer(
            'navigate',
            ControllerNavigateAction,
            execute_cb=self.navigate_callback,
            auto_start=False
        )
        self.navigate_server.start()

        self.camera_topic = '/xtion/rgb/image_raw'

        rospy.loginfo("Ready?! Vodafone!")

    def spin_callback(self, goal):
        if self.busy:
            rospy.logwarn("Me busy!!! Stop bothering!")
            self.spin_server.set_aborted()
            return

        self.busy = True

        # Python 2 compatibility: replaced f-string with .format()
        rospy.loginfo("Starting work: Take {} pictures every {}.".format(goal.num_pictures, goal.step_angle))

        step_angle_rad = math.radians(goal.step_angle)
        angular_speed = 0.5
        rotation_time = step_angle_rad / angular_speed

        rate = rospy.Rate(10)

        feedback = ControllerSpinFeedback()

        for i in range(goal.num_pictures):
            if self.spin_server.is_preempt_requested():
                self.spin_server.set_preempted()
                rospy.loginfo("Spin preempted!")
                break

            # Pause for 1 second to let the camera physically stabilize before the next shot
            rospy.sleep(1.0)

            current_angle_deg = i * goal.step_angle
            rospy.loginfo("Taking picture {}/{} at {} degrees...".format(i+1, goal.num_pictures, current_angle_deg))

            try:
                image = rospy.wait_for_message(self.camera_topic, Image, timeout=5.0)

                feedback.image_id = i
                feedback.image_data = image
                self.spin_server.publish_feedback(feedback)

                rospy.loginfo("Feedback id={} published".format(i))

            except rospy.ROSException:
                rospy.logwarn("Timeout! Failed to get image from {}".format(self.camera_topic))


            # TODO: Implement more accurate rotation using odometry
            rospy.loginfo("Rotating...")
            vel_msg = Twist()
            vel_msg.angular.z = angular_speed
            self.cmd_pub.publish(vel_msg)

            start_time = rospy.Time.now()
            while (rospy.Time.now() - start_time).to_sec() < rotation_time and not rospy.is_shutdown():
                self.cmd_pub.publish(vel_msg)
                rate.sleep()

            # Stop the robot securely by publishing an empty Twist (all zeros)
            self.cmd_pub.publish(Twist())

        rospy.loginfo("Work is done here! Spinned enough! Me no busy!")

        self.busy = False
        self.spin_server.set_succeeded()

    def navigate_callback(self, goal):
        if self.busy:
            rospy.logwarn("Me busy!!! Stop bothering!")
            self.navigate_server.set_aborted()
            return

        self.busy = True

        # Python 2 compatibility: replaced f-string with .format()
        rospy.loginfo("Starting work: Sailing to [{}, {}].".format(goal.x, goal.y))

        # TODO: Implement navigation
        rospy.sleep(1.0)

        rospy.loginfo("Work is done here! Got sea sick! Me no busy!")

        # TODO: We should abort if we can't reach the destination
        self.navigate_server.set_succeeded()


if __name__ == '__main__':
    rospy.init_node('controller', anonymous=True)

    while rospy.Time.now() == rospy.Time(0) and not rospy.is_shutdown():
        rospy.loginfo_throttle(1.0, "Waiting for Gazebo simulation time to start...")
        rospy.sleep(0.1)

    controller = Controller()

    rospy.spin()