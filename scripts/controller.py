#!/usr/bin/env python

import threading
import rospy
import math
import actionlib
import cv_bridge
import numpy as np
import tf
import tf.transformations as tft

from geometry_msgs.msg import Twist
from sensor_msgs.msg import Image, CameraInfo
from tiago_project.msg import ControllerSpinAction, ControllerSpinFeedback
from tiago_project.msg import ControllerNavigateAction, ControllerNavigateFeedback
from tf.transformations import euler_from_quaternion
from nav_msgs.msg import Odometry

class OdomTracker:
    def __init__(self):
        self.current_yaw = None
        rospy.Subscriber('/mobile_base_controller/odom', Odometry, self.odom_callback)

    def odom_callback(self, msg):
        orientation_q = msg.pose.pose.orientation
        orientation_list = [orientation_q.x, orientation_q.y, orientation_q.z, orientation_q.w]
        (roll, pitch, yaw) = euler_from_quaternion(orientation_list)
        self.current_yaw = yaw

def normalize_angle(angle):
    while angle > math.pi:
        angle -= 2.0 * math.pi
    while angle < -math.pi:
        angle += 2.0 * math.pi
    return angle


class Controller:
    def __init__(self):
        self.state_lock = threading.Lock()
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
        self.odom_tracker = OdomTracker()

        self.camera_topic = '/xtion/rgb/image_raw'
        self.depth_topic = '/xtion/depth_registered/image_raw'
        self.camera_info_topic = '/xtion/rgb/camera_info'

        rospy.loginfo("Fetching camera intrinsics from {}...".format(self.camera_info_topic))
        info_msg = rospy.wait_for_message(self.camera_info_topic, CameraInfo, timeout=5.0)

        # Extract values from the flattened 9-element K matrix
        self.camera_info = {
            "fx": info_msg.K[0],
            "cx": info_msg.K[2],
            "fy": info_msg.K[4],
            "cy": info_msg.K[5],
            "width": info_msg.width,
            "height": info_msg.width,
        }

        rospy.loginfo("Camera intrinsics locked: fx={:.1f}, fy={:.1f}, cx={:.1f}, cy={:.1f}, width={}, height={}".format(
            self.camera_info["fx"], self.camera_info["fy"], self.camera_info["cx"], self.camera_info["cy"],
            self.camera_info["width"], self.camera_info["height"]))

        self.snapshot_memory = {}
        self.bridge = cv_bridge.CvBridge()
        self.tf_listener = tf.TransformListener()

        rospy.loginfo("Ready?! Vodafone!")

    def spin_callback(self, goal):
        with self.state_lock:
            if self.busy:
                rospy.logwarn("Me busy!!! Stop bothering!")
                self.spin_server.set_aborted()
                return
            self.busy = True

        try:
            rospy.loginfo("Starting work: Take {} pictures every {}.".format(goal.num_pictures, goal.step_angle))

            step_angle_rad = math.radians(goal.step_angle)
            max_angular_speed = 1.0
            rotation_time = step_angle_rad / max_angular_speed
            initial_yaw = self.odom_tracker.current_yaw
            rate = rospy.Rate(100)

            feedback = ControllerSpinFeedback()

            for i in range(goal.num_pictures):
                if self.spin_server.is_preempt_requested():
                    self.spin_server.set_preempted()
                    rospy.loginfo("Spin preempted!")
                    break

                # Pause for 1 second to let the camera physically stabilize before the next shot
                rospy.sleep(1.0)

                c_angle_rad = self.odom_tracker.current_yaw
                c_angle_deg = math.degrees(c_angle_rad)
                i_angle_rad = normalize_angle(initial_yaw + (i * step_angle_rad))
                i_angle_deg = math.degrees(i_angle_rad)
                error_rad = normalize_angle(c_angle_rad - i_angle_rad)
                error_deg = math.degrees(error_rad)

                rospy.loginfo("Picture {} | Target: {:.2f} deg | Actual: {:.2f} deg | Error: {:.2f} deg".format(
                    i+1, i_angle_deg, c_angle_deg, error_deg
                ))

                try:
                    image = rospy.wait_for_message(self.camera_topic, Image, timeout=5.0)
                    depth_image = rospy.wait_for_message(self.depth_topic, Image, timeout=5.0)

                    # Grab the exact transform from Map to Camera Lens
                    self.tf_listener.waitForTransform('/map', '/xtion_rgb_optical_frame', rospy.Time(0), rospy.Duration(1.0))
                    (trans, rot) = self.tf_listener.lookupTransform('/map', '/xtion_rgb_optical_frame', rospy.Time(0))

                    self.snapshot_memory[i] = {
                        'depth': depth_image,
                        'trans': trans,
                        'rot': rot
                    }

                    # TODO: maybe implement persistent image ids, so that past
                    # images from previous scans are also available
                    feedback.image_id = i
                    feedback.image_data = image
                    self.spin_server.publish_feedback(feedback)

                    rospy.loginfo("Feedback id={} published".format(i))

                except rospy.ROSException:
                    rospy.logwarn("Timeout! Failed to get image from {}".format(self.camera_topic))
                except tf.Exception as e:
                    rospy.logwarn("TF Error while taking picture: {}".format(e))

                target_yaw = normalize_angle(initial_yaw + ((i + 1) * step_angle_rad))
                vel_msg = Twist()

                while not (rospy.is_shutdown() or self.spin_server.is_preempt_requested()):
                    error = normalize_angle(target_yaw - self.odom_tracker.current_yaw)

                    if abs(error) < 0.02:
                        break

                    p_speed = 0.8 * error

                    if p_speed > 0:
                        vel_msg.angular.z = min(max(p_speed, 0.1), max_angular_speed)
                    else:
                        vel_msg.angular.z = max(min(p_speed, -0.1), -max_angular_speed)

                    self.cmd_pub.publish(vel_msg)
                    rate.sleep()

                self.cmd_pub.publish(Twist())

            rospy.loginfo("Work is done here! Spinned enough! Me no busy!")

        finally:
            with self.state_lock:
                self.busy = False
            self.spin_server.set_succeeded()

    def navigate_callback(self, goal):
        with self.state_lock:
            if self.busy:
                rospy.logwarn("Me busy!!! Stop bothering!")
                self.navigate_server.set_aborted()
                return
            self.busy = True

        try:
            rospy.loginfo("Deprojecting pixel ({}, {}) from image {}...".format(
                goal.target_u, goal.target_v, goal.image_id))

            if goal.image_id not in self.snapshot_memory:
                rospy.logerr("Memory error! I have no data for image {}.".format(goal.image_id))
                self.navigate_server.set_aborted()
                return

            snapshot = self.snapshot_memory[goal.image_id]
            trans = snapshot['trans']
            rot = snapshot['rot']

            Z = goal.depth

            fx = self.camera_info["fx"]
            fy = self.camera_info["fy"]
            cx = self.camera_info["cx"]
            cy = self.camera_info["cy"]
            img_w = self.camera_info["width"]
            img_h = self.camera_info["height"]

            pixel_u = goal.target_u * img_w
            pixel_v = goal.target_v * img_h

            X_camera = (pixel_u - cx) * Z / fx
            Y_camera = (pixel_v - cy) * Z / fy
            Z_camera = Z

            # Apply the transform
            matrix = tft.quaternion_matrix(rot)
            matrix[0:3, 3] = trans

            point_camera = np.array([X_camera, Y_camera, Z_camera, 1.0])
            point_map = np.dot(matrix, point_camera)

            target_x = point_map[0]
            target_y = point_map[1]

            # ---------------------------------------------------------

            rospy.loginfo("Starting work: Sailing to [{:.2f}, {:.2f}].".format(target_x, target_y))

            # TODO: Implement actual navigation driving here
            rospy.sleep(1.0)

            rospy.loginfo("Work is done here! Got sea sick! Me no busy!")

        finally:
            with self.state_lock:
                self.busy = False
            self.navigate_server.set_succeeded()


if __name__ == '__main__':
    rospy.init_node('controller', anonymous=True)

    while rospy.Time.now() == rospy.Time(0) and not rospy.is_shutdown():
        rospy.loginfo_throttle(1.0, "Waiting for Gazebo simulation time to start...")
        rospy.sleep(0.1)

    controller = Controller()
    rospy.spin()
