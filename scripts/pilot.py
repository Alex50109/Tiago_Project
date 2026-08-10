#!/usr/bin/env python

import rospy
import actionlib
from tiago_project.msg import ControllerSpinAction, ControllerSpinGoal
from tiago_project.msg import ControllerNavigateAction, ControllerNavigateGoal

class Pilot:
    def __init__(self):
        self.spin_client = actionlib.SimpleActionClient('spin', ControllerSpinAction)
        self.navigate_client = actionlib.SimpleActionClient('navigate', ControllerNavigateAction)

        rospy.loginfo("Pilot waking up... Waiting for controller to come online.")

        self.spin_client.wait_for_server()
        self.navigate_client.wait_for_server()

        rospy.loginfo("Controller linked. Ready to command!")

    def spin_feedback_callback(self, feedback):
        rospy.loginfo(f"Pilot received picture {feedback.image_id + 1}!")
        rospy.loginfo(f"Image data size: {len(feedback.image_data.data)} bytes.")

    def command_spin(self, num_pictures, step_angle):
        rospy.loginfo(f"Ordering spin: {num_pictures} pics every {step_angle} degrees.")

        goal = ControllerSpinGoal()
        goal.num_pictures = num_pictures
        goal.step_angle = step_angle

        self.spin_client.send_goal(goal, feedback_cb=self.spin_feedback_callback)

    def cancel_spin(self):
        rospy.loginfo(f"Cancelling the spin")
        self.spin_client.cancel_all_goals()


if __name__ == '__main__':
    rospy.init_node('pilot_interface', anonymous=True)

    pilot = Pilot()

    pilot.command_spin(num_pictures=8, step_angle=45.0)

    success = pilot.spin_client.wait_for_result()
    rospy.loginfo("Spin sequence finished {}", success)

    rospy.spin()