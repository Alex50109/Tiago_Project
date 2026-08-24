#!/usr/bin/env python

import sys
import math
import rospy
import actionlib
import tf
from move_base_msgs.msg import MoveBaseAction, MoveBaseGoal
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

def get_robot_pose(tf_listener):
    """Gets the current X, Y position of the robot."""
    try:
        tf_listener.waitForTransform('/map', '/base_footprint', rospy.Time(0), rospy.Duration(4.0))
        (trans, rot) = tf_listener.lookupTransform('/map', '/base_footprint', rospy.Time(0))
        return trans[0], trans[1]
    except Exception as e:
        rospy.logerr("Could not find robot position: " + str(e))
        return None, None

def navigate_with_standoff(target_x, target_y, standoff_dist=0.5):
    tf_listener = tf.TransformListener()
    nav_client = actionlib.SimpleActionClient('move_base', MoveBaseAction)

    rospy.loginfo("Waiting for move_base...")
    nav_client.wait_for_server()
    head_pub = rospy.Publisher('/head_controller/command', JointTrajectory, queue_size=1)
    rospy.sleep(1.0) # Let TF buffer fill

    # 1. Calculate the safe 0.5m stopping coordinate
    robot_x, robot_y = get_robot_pose(tf_listener)
    if robot_x is None:
        return

    dx = target_x - robot_x
    dy = target_y - robot_y
    distance_to_target = math.hypot(dx, dy)
    yaw_angle = math.atan2(dy, dx) # Angle pointing from robot to object

    if distance_to_target <= standoff_dist:
        rospy.loginfo("Robot is already within {}m of the target.".format(standoff_dist))
        return

    # Calculate coordinate exactly 0.5m short of the object
    goal_x = target_x - (standoff_dist * math.cos(yaw_angle))
    goal_y = target_y - (standoff_dist * math.sin(yaw_angle))

    rospy.loginfo("Object at ({}, {}). Driving to safe standoff at ({:.2f}, {:.2f})...".format(
        target_x, target_y, goal_x, goal_y))

    # 2. Send the Goal
    goal = MoveBaseGoal()
    goal.target_pose.header.frame_id = "map"
    goal.target_pose.header.stamp = rospy.Time.now()
    goal.target_pose.pose.position.x = goal_x
    goal.target_pose.pose.position.y = goal_y

    # Convert yaw angle so the robot is facing the object when it stops
    q = tf.transformations.quaternion_from_euler(0, 0, yaw_angle)
    goal.target_pose.pose.orientation.x = q[0]
    goal.target_pose.pose.orientation.y = q[1]
    goal.target_pose.pose.orientation.z = q[2]
    goal.target_pose.pose.orientation.w = q[3]

    nav_client.send_goal(goal)

    # 3. Lock the neck down while the wheels are moving
    rate = rospy.Rate(5)
    while not rospy.is_shutdown():
        state = nav_client.get_state()
        if state in [actionlib.GoalStatus.SUCCEEDED, actionlib.GoalStatus.ABORTED, actionlib.GoalStatus.REJECTED]:
            break

        traj = JointTrajectory()
        traj.joint_names = ['head_1_joint', 'head_2_joint']
        point = JointTrajectoryPoint()
        point.positions = [0.0, -0.6] # Look forward, tilt down
        point.time_from_start = rospy.Duration(0.2)
        traj.points.append(point)

        head_pub.publish(traj)
        rate.sleep()

    # 4. Report Status
    if state == actionlib.GoalStatus.SUCCEEDED:
        rospy.loginfo("SUCCESS: Arrived 0.5m away from the object!")
    else:
        rospy.logwarn("FAILED: Target is blocked or unreachable (State code: {}).".format(state))


if __name__ == '__main__':
    try:
        rospy.init_node('real_pathfinder', anonymous=True)

        if len(sys.argv) < 3:
            rospy.logerr("Usage: rosrun my_pkg real_pathfinder.py <object_X> <object_Y>")
            sys.exit(1)

        OBJ_X = float(sys.argv[1])
        OBJ_Y = float(sys.argv[2])

        navigate_with_standoff(OBJ_X, OBJ_Y, standoff_dist=0.5)

    except ValueError:
        rospy.logerr("Coordinates must be numbers.")
    except rospy.ROSInterruptException:
        pass