#!/usr/bin/env python

import queue
import base64
import json
import cv2
import rospy
import actionlib
import cv_bridge
import urllib2 as urllib_req
import zlib
import numpy as np

from tiago_project.msg import ControllerSpinAction, ControllerSpinGoal
from tiago_project.msg import ControllerNavigateAction, ControllerNavigateGoal
from tiago_project.prompts import prompt_instruction_parser, prompt_object_detection

SERVER_IP = "192.168.1.102"

VLM_API_URL = "http://{}:8000/v1/chat/completions".format(SERVER_IP)
DEPTH_SERVER_URL = "http://{}:9000/predict_depth_raw".format(SERVER_IP)

SPIN_STEP_ANGLE = 45.0
SPIN_STEPS = 8
QUEUE_SIZE = 8
VLM_IMAGE_SIZE = (1280.0, 720.0)

class Pilot:
    def __init__(self):
        self.spin_client = actionlib.SimpleActionClient('spin', ControllerSpinAction)
        self.navigate_client = actionlib.SimpleActionClient('navigate', ControllerNavigateAction)

        rospy.loginfo("Pilot waking up... Waiting for controller to come online.")

        self.spin_client.wait_for_server()
        self.navigate_client.wait_for_server()

        rospy.loginfo("Controller linked. Ready to command!")

        self.is_sim = rospy.get_param('/use_sim_time', False)

        if self.is_sim:
            rospy.loginfo("I am running in a simulation (or playing a rosbag).")
        else:
            rospy.loginfo("I am running on the real physical robot.")

        self.task_prompt = None
        self.done_spinning = False
        self.targets = []

        self.spin_image_queue = queue.Queue(SPIN_STEPS)

    # TODO: can be structured as a service
    def execute_task(self, instructions):
        target = parse_instructions_prompt(instructions)
        if target is None:
            rospy.logerr("Not an interesting command! I have the right to ignore you!")
            return False

        rospy.loginfo("Command is to go to %s (keep %s in mind)!", target["target"], target["desc"])

        # clear queue (maybe there is a better way)
        while True:
            try:
                self.spin_image_queue.get_nowait()
            except queue.Empty as e:
                break

        # start spinning
        self.spin_client.send_goal(ControllerSpinGoal(num_pictures=SPIN_STEPS, step_angle=SPIN_STEP_ANGLE), feedback_cb=self.spin_feedback_callback)

        images_processed = 0
        found_target = None
        found_image_id = None
        found_encoded_image = None

        while images_processed < SPIN_STEPS and not rospy.is_shutdown():
            try:
                image_id, image_data = self.spin_image_queue.get(timeout=1.0)
                images_processed += 1

                encoded_image = encode_image(image_data)

                rospy.loginfo("Prompting picture %d!", image_id + 1)
                result = detect_targets_in_image(target, encoded_image)
                rospy.loginfo("LLM returned: %s", result)

                if len(result) > 0:
                    found_image_id, found_target = (image_id, result[0])
                    rospy.loginfo("Target found! Canceling remaining spin.")

                    current_state = self.spin_client.get_state()
                    if current_state in [actionlib.GoalStatus.PENDING, actionlib.GoalStatus.ACTIVE]:
                        rospy.loginfo("Canceling the remaining spin.")
                        self.spin_client.cancel_goal()
                    else:
                        rospy.loginfo("Spin was already finished. No need to cancel.")

                    break

            except queue.Empty:
                # If we waited 1 second and got no picture, check if the action server stopped
                state = self.spin_client.get_state()

                # If the server succeeded, aborted, or was preempted, it won't send more images
                if state in [actionlib.GoalStatus.SUCCEEDED, actionlib.GoalStatus.ABORTED, actionlib.GoalStatus.PREEMPTED]:
                    rospy.logwarn("Spin action ended early. Stopping image processing.")
                    break

                # Otherwise, it's just taking a while. Loop around and wait again.
                continue

        if found_target is None:
            rospy.logerr("Target not found!")
            return False

        x_min, y_min, x_max, y_max = found_target["box"]

        true_x_min = min(max(x_min / 1000.0, 0.0), 1.0)
        true_y_min = min(max(y_min / 1000.0, 0.0), 1.0)
        true_x_max = min(max(x_max / 1000.0, 0.0), 1.0)
        true_y_max = min(max(y_max / 1000.0, 0.0), 1.0)

        center_u = (true_x_min + true_x_max) / 2
        center_v = (true_y_min + true_y_max) / 2

        rospy.loginfo("Object center is at coords: u={}, v={}".format(center_u, center_v))

        if not self.is_sim:
            depth_data = get_depth_data(found_encoded_image)

            # TODO: get depth and send it
            depth_h, depth_w = depth_data.shape
            pixel_x = int(center_u * (depth_w - 1))
            pixel_y = int(center_v * (depth_h - 1))

            # Sample a 5x5 window and take median to avoid noise/outliers
            y_start = max(0, pixel_y - 2)
            y_end = min(depth_h, pixel_y + 3)
            x_start = max(0, pixel_x - 2)
            x_end = min(depth_w, pixel_x + 3)

            patch = depth_data[y_start:y_end, x_start:x_end]
            depth = float(np.nanmedian(patch))
        else:
            # if we are in the simulator we don't need to care about longer distances
            depth = 0.0

        rospy.loginfo("Object center is at {:.2f} meters distance".format(depth))

        # Wait for spinning to finish
        self.spin_client.wait_for_result()

        # Sleep a bit to ensure the controller is not busy (a bit hacky)
        rospy.sleep(0.5)

        self.navigate_client.send_goal_and_wait(ControllerNavigateGoal(
            target_u = center_u,
            target_v = center_v,
            depth = depth,
            image_id=found_image_id
        ))

        return True

    def spin_feedback_callback(self, feedback):
        rospy.loginfo("Pilot received picture %d!", feedback.image_id + 1)
        rospy.loginfo("Image data size: %d bytes.", len(feedback.image_data.data))

        self.spin_image_queue.put((feedback.image_id, feedback.image_data))

bridge = cv_bridge.CvBridge()

def encode_image(img):
    try:
        # Convert ROS image to OpenCV BGR image (standard for cv2)
        cv_img = bridge.imgmsg_to_cv2(img, desired_encoding="bgr8")
    except cv_bridge.CvBridgeError as e:
        rospy.logerr("CvBridge Error: %s" % str(e))
        return None

    h, w = cv_img.shape[:2]
    max_w, max_h = VLM_IMAGE_SIZE

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

    return raw_bytes

def prompt_model(text, image):
    content = [{"type": "text", "text": text}]

    if image is not None:
        base64_image = base64.b64encode(image)

        # Ensure it's returned as a unicode string on both Python 2 and 3
        if isinstance(base64_image, bytes):
            base64_image = base64_image.decode("utf-8")

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
        "max_tokens": 1000,
        "temperature": 0.2
    }

    headers = {
        "Content-Type": "application/json",
        "Authorization": "Bearer EMPTY"  # Required by some local servers even if auth is off
    }

    # Encode JSON payload to bytes for HTTP request payload compatibility
    json_data = json.dumps(payload).encode("utf-8")

    # Create the HTTP request
    request = urllib_req.Request(VLM_API_URL, data=json_data, headers=headers)

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
    reply = prompt_model(prompt_instruction_parser.format(prompt), None)

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

def detect_targets_in_image(instructions, image):
    reply = prompt_model(prompt_object_detection.format(instructions["target"], instructions["desc"]), image)
    try:
        items = []
        image_width, image_height = VLM_IMAGE_SIZE
        for item in json.loads(reply.strip()):
            box = item["box"]

            items.append({
                "desc": item["desc"],
                "box":  box,
            })
        return items

    except ValueError as e:
        # This catches JSON parsing errors if the LLM hallucinates bad formatting
        rospy.logerr("Failed to parse JSON from LLM. Raw output was: %s", reply)
        return None

def get_depth_data(image):
    req = urllib_req.Request(DEPTH_SERVER_URL, data=image)
    req.add_header('Content-Type', 'application/octet-stream')
    req.add_header('Content-Length', str(len(image)))

    response = urllib_req.urlopen(req)

    h = int(response.headers.get('X-Depth-Height'))
    w = int(response.headers.get('X-Depth-Width'))

    compressed_data = response.read()
    raw_bytes = zlib.decompress(compressed_data)

    depth_map = np.fromstring(raw_bytes, dtype=np.float32).reshape((h, w))

    return depth_map

if __name__ == '__main__':
    rospy.init_node('pilot_interface', anonymous=True)

    pilot = Pilot()
    pilot.execute_task("Go to the bench next to a person!")

    rospy.spin()
