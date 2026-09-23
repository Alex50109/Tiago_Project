#!/usr/bin/env python

import threading
import rospy
import math
import actionlib
import cv_bridge
import message_filters
import numpy as np
import tf
import tf.transformations as tft

from geometry_msgs.msg import Twist, Point
from sensor_msgs.msg import Image, CameraInfo
from tiago_project.msg import ControllerFindObjectAction, ControllerFindObjectFeedback, ControllerFindObjectResult
from tiago_project.msg import TargetUpdate
from tf.transformations import euler_from_quaternion
from nav_msgs.msg import Odometry
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from move_base_msgs.msg import MoveBaseAction, MoveBaseGoal
from std_srvs.srv import Empty

HEAD_TILT = -0.1
HEAD_PAN_LIMIT = math.radians(70.0)
HEAD_PAN_STEP = math.radians(2.0)
STANDOFF_DIST = 0.5

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
        self.update_lock = threading.Lock()
        self.busy = False

        self.regoal_threshold = rospy.get_param('~regoal_threshold', 0.3)
        self.update_wait_timeout = rospy.get_param('~update_wait_timeout', 10.0)
        self.capture_period = rospy.get_param('~capture_period', 1.0)

        self.cmd_pub = rospy.Publisher('/mobile_base_controller/cmd_vel', Twist, queue_size=10)
        self.head_pub = rospy.Publisher('/head_controller/command', JointTrajectory, queue_size=1)

        self.latest_update = None
        self.update_sub = rospy.Subscriber('/target_update', TargetUpdate, self.target_update_callback)

        self.find_server = actionlib.SimpleActionServer(
            'find_object',
            ControllerFindObjectAction,
            execute_cb=self.find_callback,
            auto_start=False
        )
        self.find_server.start()
        self.odom_tracker = OdomTracker()

        self.nav_client = actionlib.SimpleActionClient('move_base', MoveBaseAction)
        rospy.loginfo("Waiting for move_base...")
        self.nav_client.wait_for_server()

        self.clear_costmaps_srv = rospy.ServiceProxy('/move_base/clear_costmaps', Empty)

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
            "height": info_msg.height,
        }

        rospy.loginfo("Camera intrinsics locked: fx={:.1f}, fy={:.1f}, cx={:.1f}, cy={:.1f}, width={}, height={}".format(
            self.camera_info["fx"], self.camera_info["fy"], self.camera_info["cx"], self.camera_info["cy"],
            self.camera_info["width"], self.camera_info["height"]))

        self.snapshot_memory = {}
        self.next_image_id = 0
        self.bridge = cv_bridge.CvBridge()
        self.tf_listener = tf.TransformListener()

        self.frame_lock = threading.Lock()
        self.latest_frame = None
        self.rgb_sub = message_filters.Subscriber(self.camera_topic, Image)
        self.depth_sub = message_filters.Subscriber(self.depth_topic, Image)
        self.frame_sync = message_filters.ApproximateTimeSynchronizer(
            [self.rgb_sub, self.depth_sub], queue_size=5, slop=0.1)
        self.frame_sync.registerCallback(self.frame_sync_callback)

        rospy.loginfo("Ready?! Vodafone!")

    def frame_sync_callback(self, rgb_msg, depth_msg):
        with self.frame_lock:
            self.latest_frame = (rgb_msg, depth_msg)

    def target_update_callback(self, msg):
        with self.update_lock:
            self.latest_update = msg

    def take_latest_update(self):
        with self.update_lock:
            update = self.latest_update
            self.latest_update = None
        return update

    def get_target_map_point(self, image_id, u, v, depth=1.0):
        """
        Deprojects a 2D pixel from a past snapshot into a 3D Map coordinate.
        If depth is omitted, it defaults to 1.0 (useful for ray-casting / yaw alignment).
        """
        if image_id not in self.snapshot_memory:
            rospy.logerr("Memory error! I have no data for image {}.".format(image_id))
            return None, None

        snapshot = self.snapshot_memory[image_id]
        trans = snapshot['trans']
        rot = snapshot['rot']

        fx = self.camera_info["fx"]
        fy = self.camera_info["fy"]
        cx = self.camera_info["cx"]
        cy = self.camera_info["cy"]
        img_w = self.camera_info["width"]
        img_h = self.camera_info["height"]

        pixel_u = u * img_w
        pixel_v = v * img_h

        X_camera = (pixel_u - cx) * depth / fx
        Y_camera = (pixel_v - cy) * depth / fy
        Z_camera = depth

        # Apply the transform
        matrix = tft.quaternion_matrix(rot)
        matrix[0:3, 3] = trans

        point_camera = np.array([X_camera, Y_camera, Z_camera, 1.0])
        point_map = np.dot(matrix, point_camera)

        return point_map[0], point_map[1]

    def rotate_to_yaw(self, target_yaw):
        """
        Rotates the robot base to a specific yaw angle using proportional control.
        """
        rate = rospy.Rate(100)
        vel_msg = Twist()
        max_angular_speed = 1.0

        while not (rospy.is_shutdown() or self.find_server.is_preempt_requested()):
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

        self.cmd_pub.publish(Twist())  # Stop moving once done

    def set_head_pose(self, pan, tilt):
        traj = JointTrajectory()
        traj.joint_names = ['head_1_joint', 'head_2_joint']
        point = JointTrajectoryPoint()
        point.positions = [pan, tilt]
        point.time_from_start = rospy.Duration(0.2)
        traj.points.append(point)
        self.head_pub.publish(traj)

    def aim_head_at_map_point(self, target_x, target_y, current_pan):
        """
        Points the head pan at a map coordinate, clamped to the head joint range.
        Returns the pan actually applied so changes can be throttled across calls.
        """
        robot_x, robot_y, robot_yaw = self.get_robot_pose()
        if robot_x is None:
            return current_pan

        bearing = math.atan2(target_y - robot_y, target_x - robot_x)
        pan = max(min(normalize_angle(bearing - robot_yaw), HEAD_PAN_LIMIT), -HEAD_PAN_LIMIT)

        if abs(normalize_angle(pan - current_pan)) >= HEAD_PAN_STEP:
            self.set_head_pose(pan, HEAD_TILT)
            return pan
        return current_pan

    def get_robot_pose(self):
        """Gets the current X, Y position and yaw of the robot in the map frame."""
        try:
            self.tf_listener.waitForTransform('/map', '/base_footprint', rospy.Time(0), rospy.Duration(4.0))
            (trans, rot) = self.tf_listener.lookupTransform('/map', '/base_footprint', rospy.Time(0))
            yaw = euler_from_quaternion(rot)[2]
            return trans[0], trans[1], yaw
        except Exception as e:
            rospy.logerr("Could not find robot position: " + str(e))
            return None, None, None

    def capture_frame(self):
        """
        Grabs the newest time-synchronized RGB+depth pair and stores the map->camera
        snapshot for the RGB timestamp, which is also the depth frame's reference.
        """
        deadline = rospy.Time.now() + rospy.Duration(5.0)
        rate = rospy.Rate(20)
        frame = None

        while not rospy.is_shutdown() and frame is None:
            with self.frame_lock:
                frame = self.latest_frame
                self.latest_frame = None

            if frame is None:
                if rospy.Time.now() > deadline:
                    raise rospy.ROSException("Timeout! No synchronized RGB+depth pair arrived.")
                rate.sleep()

        if frame is None:
            raise rospy.ROSException("Shutting down while waiting for a synchronized frame.")

        image, depth_image = frame
        img_time = image.header.stamp
        self.tf_listener.waitForTransform('/map', '/xtion_rgb_optical_frame', img_time, rospy.Duration(1.0))
        (trans, rot) = self.tf_listener.lookupTransform('/map', '/xtion_rgb_optical_frame', img_time)

        image_id = self.next_image_id
        self.next_image_id += 1
        self.snapshot_memory[image_id] = {
            'trans': trans,
            'rot': rot
        }

        return image_id, image, depth_image

    def publish_image_feedback(self, image_id, image, depth_image, navigating):
        feedback = ControllerFindObjectFeedback()
        feedback.image_id = image_id
        feedback.image_data = image
        feedback.depth_data = depth_image
        feedback.navigating = navigating
        self.find_server.publish_feedback(feedback)
        rospy.loginfo("Feedback id={} published".format(image_id))

    def build_move_base_goal(self, target_x, target_y, robot_x, robot_y):
        """Builds a move_base goal at the standoff point in front of the target, facing it."""
        dx = target_x - robot_x
        dy = target_y - robot_y
        yaw_angle = math.atan2(dy, dx)

        goal_x = target_x - (STANDOFF_DIST * math.cos(yaw_angle))
        goal_y = target_y - (STANDOFF_DIST * math.sin(yaw_angle))

        rospy.loginfo("Object at ({:.2f}, {:.2f}). Driving to safe standoff at ({:.2f}, {:.2f})...".format(
            target_x, target_y, goal_x, goal_y))

        nav_goal = MoveBaseGoal()
        nav_goal.target_pose.header.frame_id = "map"
        nav_goal.target_pose.header.stamp = rospy.Time.now()
        nav_goal.target_pose.pose.position.x = goal_x
        nav_goal.target_pose.pose.position.y = goal_y

        q = tft.quaternion_from_euler(0, 0, yaw_angle)
        nav_goal.target_pose.pose.orientation.x = q[0]
        nav_goal.target_pose.pose.orientation.y = q[1]
        nav_goal.target_pose.pose.orientation.z = q[2]
        nav_goal.target_pose.pose.orientation.w = q[3]

        return nav_goal

    def scan_phase(self, goal):
        """
        Rotates in place and streams pictures to the pilot.
        Returns (status, target_point): the point comes from the first pilot update,
        ending the scan early only when navigation was requested.
        """
        detected_point = None
        step_angle_rad = math.radians(goal.step_angle)
        initial_yaw = self.odom_tracker.current_yaw

        for i in range(goal.num_pictures):
            if self.find_server.is_preempt_requested():
                rospy.loginfo("Scan preempted!")
                return 'preempted', None

            # Pause for 1 second to let the camera physically stabilize before the next shot
            rospy.sleep(1.0)

            c_angle_rad = self.odom_tracker.current_yaw
            c_angle_deg = math.degrees(c_angle_rad)
            i_angle_rad = normalize_angle(initial_yaw + (i * step_angle_rad))
            i_angle_deg = math.degrees(i_angle_rad)
            error_deg = math.degrees(normalize_angle(c_angle_rad - i_angle_rad))

            rospy.loginfo("Picture {} | Target: {:.2f} deg | Actual: {:.2f} deg | Error: {:.2f} deg".format(
                i + 1, i_angle_deg, c_angle_deg, error_deg
            ))

            try:
                image_id, image, depth_image = self.capture_frame()
                self.publish_image_feedback(image_id, image, depth_image, navigating=False)
            except rospy.ROSException:
                rospy.logwarn("Timeout! Failed to get image from {}".format(self.camera_topic))
            except tf.Exception as e:
                rospy.logwarn("TF Error while taking picture: {}".format(e))

            update = self.take_latest_update()
            if update is not None:
                point = self.get_target_map_point(update.image_id, update.target_u, update.target_v, depth=update.depth)
                if point[0] is not None:
                    rospy.loginfo("Target reported by pilot at image {}!".format(update.image_id))
                    detected_point = point
                    if goal.navigate:
                        return 'ok', detected_point
                else:
                    rospy.logwarn("Target update referenced unknown image {}.".format(update.image_id))

            if i < goal.num_pictures - 1:
                next_yaw = normalize_angle(initial_yaw + ((i + 1) * step_angle_rad))
                self.rotate_to_yaw(next_yaw)

        return 'ok', detected_point

    def wait_for_first_update(self, timeout):
        """
        Gives the pilot time to report a late detection after the scan has ended.
        Returns (status, target_point).
        """
        rospy.loginfo("Scan done. Waiting up to {:.0f}s for the pilot to report a detection...".format(timeout))
        deadline = rospy.Time.now() + rospy.Duration(timeout)
        rate = rospy.Rate(10)

        while not rospy.is_shutdown() and rospy.Time.now() < deadline:
            if self.find_server.is_preempt_requested():
                rospy.loginfo("Find preempted while waiting for a detection!")
                return 'preempted', None

            update = self.take_latest_update()
            if update is not None:
                point = self.get_target_map_point(update.image_id, update.target_u, update.target_v, depth=update.depth)
                if point[0] is not None:
                    return 'ok', point
                rospy.logwarn("Target update referenced unknown image {}.".format(update.image_id))

            rate.sleep()

        return 'ok', None

    def navigate_phase(self, target_point):
        """
        Drives to the standoff point in front of the target, streaming pictures and
        re-goaling move_base whenever pilot corrections shift the target far enough.
        Returns (status, target_point, reached).
        """
        reached = False
        robot_x, robot_y, _ = self.get_robot_pose()
        if robot_x is None:
            return 'aborted', target_point, reached

        if math.hypot(target_point[0] - robot_x, target_point[1] - robot_y) <= STANDOFF_DIST:
            rospy.loginfo("Robot is already within {}m of the target.".format(STANDOFF_DIST))
            return 'ok', target_point, True

        nav_goal = self.build_move_base_goal(target_point[0], target_point[1], robot_x, robot_y)
        self.nav_client.send_goal(nav_goal)
        goal_point = (nav_goal.target_pose.pose.position.x, nav_goal.target_pose.pose.position.y)

        head_pan = 0.0
        last_capture_time = rospy.Time(0)
        rate = rospy.Rate(5)

        while not rospy.is_shutdown():
            if self.find_server.is_preempt_requested():
                rospy.loginfo("Navigation preempted! Canceling move_base goal.")
                self.nav_client.cancel_goal()
                return 'preempted', target_point, reached

            state = self.nav_client.get_state()
            if state in [actionlib.GoalStatus.SUCCEEDED, actionlib.GoalStatus.ABORTED, actionlib.GoalStatus.REJECTED]:
                break

            update = self.take_latest_update()
            if update is not None:
                corrected = self.get_target_map_point(update.image_id, update.target_u, update.target_v, depth=update.depth)
                if corrected[0] is not None:
                    target_point = corrected
                    robot_x, robot_y, _ = self.get_robot_pose()
                    if robot_x is not None:
                        if math.hypot(target_point[0] - robot_x, target_point[1] - robot_y) <= STANDOFF_DIST:
                            rospy.loginfo("Corrected target is within {}m. Stopping here.".format(STANDOFF_DIST))
                            self.nav_client.cancel_goal()
                            return 'ok', target_point, True

                        shift = math.hypot(target_point[0] - goal_point[0], target_point[1] - goal_point[1])
                        if shift > self.regoal_threshold:
                            rospy.loginfo("Target shifted by {:.2f}m! Re-goal!".format(shift))
                            nav_goal = self.build_move_base_goal(target_point[0], target_point[1], robot_x, robot_y)
                            self.nav_client.cancel_goal()
                            self.nav_client.send_goal(nav_goal)
                            goal_point = (nav_goal.target_pose.pose.position.x, nav_goal.target_pose.pose.position.y)
                else:
                    rospy.logwarn("Target update referenced unknown image {}.".format(update.image_id))

            head_pan = self.aim_head_at_map_point(target_point[0], target_point[1], head_pan)

            now = rospy.Time.now()
            if (now - last_capture_time).to_sec() >= self.capture_period:
                last_capture_time = now
                try:
                    image_id, image, depth_image = self.capture_frame()
                    self.publish_image_feedback(image_id, image, depth_image, navigating=True)
                except rospy.ROSException:
                    rospy.logwarn("Timeout! Failed to get image from {}".format(self.camera_topic))
                except tf.Exception as e:
                    rospy.logwarn("TF Error while taking picture: {}".format(e))

            rate.sleep()

        state = self.nav_client.get_state()
        if state == actionlib.GoalStatus.SUCCEEDED:
            rospy.loginfo("SUCCESS: Arrived {}m away from the object!".format(STANDOFF_DIST))
            return 'ok', target_point, True

        rospy.logwarn("FAILED: Target is blocked or unreachable (State code: {}).".format(state))
        return 'aborted', target_point, reached

    def find_callback(self, goal):
        with self.state_lock:
            if self.busy:
                rospy.logwarn("Me busy!!! Stop bothering!")
                self.find_server.set_aborted()
                return
            self.busy = True

        try:
            with self.update_lock:
                self.latest_update = None
            self.snapshot_memory = {}
            self.next_image_id = 0

            self.clear_costmaps_srv()
            self.set_head_pose(0.0, HEAD_TILT)

            rospy.loginfo("Starting work: {} pictures every {} deg, navigate={}.".format(
                goal.num_pictures, goal.step_angle, goal.navigate))

            status, target_point = self.scan_phase(goal)
            if status == 'preempted':
                self.find_server.set_preempted()
                return

            if goal.navigate and target_point is None:
                status, target_point = self.wait_for_first_update(self.update_wait_timeout)
                if status == 'preempted':
                    self.find_server.set_preempted()
                    return

            if goal.navigate and target_point is not None:
                status, target_point, reached = self.navigate_phase(target_point)
                if status == 'preempted':
                    self.find_server.set_preempted()
                    return
                if status == 'aborted':
                    result = ControllerFindObjectResult()
                    result.target_found = True
                    result.reached = False
                    result.final_position = Point(target_point[0], target_point[1], 0.0)
                    self.find_server.set_aborted(result)
                    return
                reached_final = reached
            else:
                reached_final = False

            rospy.loginfo("Work is done here! Me no busy!")

            result = ControllerFindObjectResult()
            result.target_found = target_point is not None
            result.reached = reached_final
            if target_point is not None:
                result.final_position = Point(target_point[0], target_point[1], 0.0)
            self.find_server.set_succeeded(result)

        finally:
            with self.state_lock:
                self.busy = False


if __name__ == '__main__':
    rospy.init_node('controller', anonymous=True)

    while rospy.Time.now() == rospy.Time(0) and not rospy.is_shutdown():
        rospy.loginfo_throttle(1.0, "Waiting for Gazebo simulation time to start...")
        rospy.sleep(0.1)

    controller = Controller()
    rospy.spin()
