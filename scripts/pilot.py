#!/usr/bin/env python

import sys
import base64
import json
import cv2
import rospy
import actionlib
import cv_bridge

# Python 2 vs Python 3 URL library compatibility
try:
    import urllib.request as urllib_req
    import urllib.error as urllib_err
except ImportError:
    import urllib2 as urllib_req
    import urllib2 as urllib_err

from tiago_project.msg import ControllerSpinAction, ControllerSpinGoal
from tiago_project.msg import ControllerNavigateAction, ControllerNavigateGoal
from tiago_project.prompts import prompt_instruction_parser

API_URL = "http://10.41.2.68:8000/v1/chat/completions"

class Pilot:
    def __init__(self):
        self.spin_client = actionlib.SimpleActionClient('spin', ControllerSpinAction)
        self.navigate_client = actionlib.SimpleActionClient('navigate', ControllerNavigateAction)

        rospy.loginfo("Pilot waking up... Waiting for controller to come online.")

        self.spin_client.wait_for_server()
        self.navigate_client.wait_for_server()

        rospy.loginfo("Controller linked. Ready to command!")

        self.task_prompt = None
        self.done_spinning = False
        self.targets = []

    # TODO: can be structured as a service
    def execute_task(self, instructions):
        # self.task_prompt = parse_instructions_prompt(instructions)
        # if not self.task_prompt:
        #     rospy.logerr("Not an interesting command! I have the right to ignore you!")
        #     return False

        self.done_spinning = False
        self.targets = []

        # start spinning
        self.spin_client.send_goal(ControllerSpinGoal(num_pictures=8, step_angle=45.0), feedback_cb=self.spin_feedback_callback)
        if not self.spin_client.wait_for_result():
            rospy.logerr("Spin failed! Aborting command!")
            return False

        while not self.done_spinning:
            rospy.sleep(1.0)

        # max_score = -1
        # max_target = None
        # for target in self.targets:
        #     if target.score > max_score:
        #         max_score = target.score
        #         max_target = target.coord

        # if max_target is None:
        #     rospy.logerr("TODO: No target found!")
        #     return False

        # self.navigate_client.send_goal_and_wait(ControllerNavigateGoal(x = max_target[0], y = max_target[1]))
        # if not self.spin_client.wait_for_result():
        #     rospy.logerr("Failed to navigate to destination!")
        #     return False

        return True

    def spin_feedback_callback(self, feedback):
        rospy.loginfo("Pilot received picture %d!", feedback.image_id + 1)
        rospy.loginfo("Image data size: %d bytes.", len(feedback.image_data.data))

        result = prompt_model("Write a sentence that best describes what is in this image", feedback.image_data)
        rospy.loginfo("LLM returned: %s", result)

bridge = cv_bridge.CvBridge()

def encode_image(img):
    try:
        # Convert ROS image to OpenCV BGR image (standard for cv2)
        cv_img = bridge.imgmsg_to_cv2(img, desired_encoding="bgr8")
    except cv_bridge.CvBridgeError as e:
        rospy.logerr("CvBridge Error: %s" % str(e))
        return None

    h, w = cv_img.shape[:2]
    max_w, max_h = 1280.0, 720.0

    scale = min(max_w / w, max_h / h)

    if scale < 1.0:
        new_size = (int(w * scale), int(h * scale))
        cv_img = cv2.resize(cv_img, new_size, interpolation=cv2.INTER_AREA)

    encode_params = [int(cv2.IMWRITE_JPEG_QUALITY), 85]
    success, buffer = cv2.imencode('.jpg', cv_img, encode_params)

    if not success:
        rospy.logerr("Failed to encode image to JPEG")
        return None

    # Handle buffer.tobytes() for newer OpenCV vs buffer.tostring() for older OpenCV versions
    if hasattr(buffer, 'tobytes'):
        raw_bytes = buffer.tobytes()
    else:
        raw_bytes = buffer.tostring()

    encoded = base64.b64encode(raw_bytes)
    
    # Ensure it's returned as a unicode string on both Python 2 and 3
    if isinstance(encoded, bytes):
        return encoded.decode("utf-8")
    return encoded

def prompt_model(text, image):
    content = [{"type": "text", "text": text}]

    if image is not None:
        base64_image = encode_image(image)
        content.append({
            "type": "image_url",
            "image_url": {
                "url": "data:image/jpeg;base64,%s" % base64_image
            }
        })

    # Construct the JSON payload exactly as the OpenAI API expects
    payload = {
        "model": "mlx-community/Qwen3-VL-4B-Instruct-4bit",
        "messages": [
            {
                "role": "user",
                "content": content,
            }
        ],
        "max_tokens": 300,
        "temperature": 0.2
    }

    headers = {
        "Content-Type": "application/json",
        "Authorization": "Bearer EMPTY"  # Required by some local servers even if auth is off
    }

    # Encode JSON payload to bytes for HTTP request payload compatibility
    json_data = json.dumps(payload).encode("utf-8")

    # Create the HTTP request
    request = urllib_req.Request(API_URL, data=json_data, headers=headers)

    try:
        # Send the request and read the response
        response = urllib_req.urlopen(request)
        response_data = response.read()

        if isinstance(response_data, bytes):
            response_data = response_data.decode("utf-8")

        # Parse the JSON string back into a Python dictionary
        result = json.loads(response_data)

        # Extract the content from the response tree
        return result["choices"][0]["message"]["content"]

    except Exception as e:
        # Catch the exception and print the actual error message to the ROS console
        rospy.logerr("Failed to query model: %s" % str(e))
        return ""

def parse_instructions_prompt(prompt):
    reply = prompt_model(prompt_instruction_parser + prompt, None)

    try:
        command_data = json.loads(reply.strip())

        if command_data.get("is_navigation") == True:
            target = command_data.get("location")
            desc = command_data.get("description")

            rospy.loginfo("Navigation command received!")
            rospy.loginfo("Target: %s", target)
            rospy.loginfo("Details: %s", desc)

            return {
                "target": target,
                "desc": desc,
            }
        else:
            rospy.loginfo("Command was parsed, but it is not a navigation task.")

    except ValueError as e:
        # This catches JSON parsing errors if the LLM hallucinates bad formatting
        rospy.logerr("Failed to parse JSON from LLM. Raw output was: %s", reply)

    return None

if __name__ == '__main__':
    rospy.init_node('pilot_interface', anonymous=True)

    pilot = Pilot()
    pilot.execute_task("Go to the board!")

    rospy.spin()