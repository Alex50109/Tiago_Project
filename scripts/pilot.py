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
from tiago_project.algorithms import ransac_linear_regression

SERVER_IP = "192.168.1.102"

VLM_API_URL = "http://{}:8000/v1/chat/completions".format(SERVER_IP)

# Port 9090 for the Non-Metric Model, Port 9000 for the OLD Metric Model
DEPTH_SERVER_URL = "http://{}:9090/predict_depth_raw".format(SERVER_IP)
METRIC_DEPTH_SERVER_URL = "http://{}:9000/predict_depth_raw".format(SERVER_IP)

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

        self.task_prompt = None
        self.done_spinning = False
        self.targets = []

        self.spin_image_queue = queue.Queue(SPIN_STEPS)

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
            except queue.Empty:
                break

        # start spinning
        self.spin_client.send_goal(ControllerSpinGoal(num_pictures=SPIN_STEPS, step_angle=SPIN_STEP_ANGLE), feedback_cb=self.spin_feedback_callback)

        images_processed = 0
        found_target = None
        found_image_id = None
        found_encoded_image = None
        found_depth_msg = None

        while images_processed < SPIN_STEPS and not rospy.is_shutdown():
            try:
                image_id, image_data, depth_data = self.spin_image_queue.get(timeout=1.0)
                images_processed += 1

                encoded_image = encode_image(image_data)

                rospy.loginfo("Prompting picture %d!", image_id + 1)
                result = detect_targets_in_image(target, encoded_image)
                rospy.loginfo("LLM returned: %s", result)

                if result is not None and len(result) > 0:
                    found_image_id = image_id
                    found_target = result[0]
                    found_encoded_image = encoded_image
                    found_depth_msg = depth_data

                    rospy.loginfo("Target found! Canceling remaining spin.")

                    current_state = self.spin_client.get_state()
                    if current_state in [actionlib.GoalStatus.PENDING, actionlib.GoalStatus.ACTIVE]:
                        rospy.loginfo("Canceling the remaining spin.")
                        self.spin_client.cancel_goal()
                    else:
                        rospy.loginfo("Spin was already finished. No need to cancel.")

                    break

            except queue.Empty:
                state = self.spin_client.get_state()
                if state in [actionlib.GoalStatus.SUCCEEDED, actionlib.GoalStatus.ABORTED, actionlib.GoalStatus.PREEMPTED]:
                    rospy.logwarn("Spin action ended early. Stopping image processing.")
                    break
                continue

        if found_target is None:
            rospy.logerr("Target not found!")
            return False

        # Extract normalized bounding box coordinates
        x_min, y_min, x_max, y_max = found_target["box"]
        u_min = min(max(x_min / 1000.0, 0.0), 1.0)
        v_min = min(max(y_min / 1000.0, 0.0), 1.0)
        u_max = min(max(x_max / 1000.0, 0.0), 1.0)
        v_max = min(max(y_max / 1000.0, 0.0), 1.0)

        center_u = (u_min + u_max) / 2.0
        center_v = (v_min + v_max) / 2.0

        rospy.loginfo("Object center is at coords: u={:.3f}, v={:.3f}".format(center_u, center_v))

        # Convert Hardware Depth to Numpy Array
        try:
            hw_depth_raw = bridge.imgmsg_to_cv2(found_depth_msg, desired_encoding="passthrough")
        except cv_bridge.CvBridgeError as e:
            rospy.logerr("CvBridge Error: %s" % str(e))
            return False

        # Convert 16UC1 (millimeters) to meters if necessary
        if hw_depth_raw.dtype == np.uint16:
            hw_depth = hw_depth_raw.astype(np.float32) / 1000.0
        else:
            hw_depth = hw_depth_raw.copy()

        hw_h, hw_w = hw_depth.shape

        # Extract Center 50% ROI of the bounding box (ROI = region of interest)
        roi_u_min = u_min + (u_max - u_min) * 0.25
        roi_u_max = u_max - (u_max - u_min) * 0.25
        roi_v_min = v_min + (v_max - v_min) * 0.25
        roi_v_max = v_max - (v_max - v_min) * 0.25

        px_min = max(0, int(roi_u_min * (hw_w - 1)))
        px_max = min(hw_w, max(px_min + 1, int(roi_u_max * (hw_w - 1))))
        py_min = max(0, int(roi_v_min * (hw_h - 1)))
        py_max = min(hw_h, max(py_min + 1, int(roi_v_max * (hw_h - 1))))

        hw_roi_patch = hw_depth[py_min:py_max, px_min:px_max]

        # Filter valid hardware pixels (e.g. valid between 0.2m and 3.0m)
        valid_hw_mask = (hw_roi_patch > 0.2) & (hw_roi_patch < 3.0) & (~np.isnan(hw_roi_patch))
        valid_hw_pixels = hw_roi_patch[valid_hw_mask]

        depth = 0.0

        # The Fusion Logic
        # If at least 5% of our ROI has valid hardware depth, use it immediately
        if len(valid_hw_pixels) > (hw_roi_patch.size * 0.05):
            depth = float(np.median(valid_hw_pixels))
            rospy.loginfo("Hardware depth acquired successfully. Object is close.")
        else:
            rospy.loginfo("Hardware depth blind in bounding box. Calibrating AI depth...")

            # Fetch AI Depth from NON-METRIC server
            ai_depth_map = get_depth_data(found_encoded_image, url=DEPTH_SERVER_URL)
            ai_depth_resized = cv2.resize(ai_depth_map, (hw_w, hw_h), interpolation=cv2.INTER_NEAREST)

            # Find background anchor points across the entire image (0.5m to 3.0m)
            anchor_mask = (hw_depth > 0.5) & (hw_depth < 3.0) & (~np.isnan(hw_depth))
            hw_anchors = hw_depth[anchor_mask]

            # INVERT THE NON-METRIC AI DEPTH TO MAKE IT METRIC-LINEAR
            ai_raw_anchors = ai_depth_resized[anchor_mask]
            ai_anchors = 1.0 / (ai_raw_anchors + 1e-6)

            # Perform RANSAC Regression
            if len(hw_anchors) > 100:
                s, t = ransac_linear_regression(ai_anchors, hw_anchors, max_iters=150, threshold=0.15)
                rospy.loginfo("Live calibration successful: scale={:.3f}, shift={:.3f}".format(s, t))
            else:
                rospy.logwarn("Not enough hardware anchors for regression. Defaulting to scale=1.0, shift=0.0")
                s, t = 1.0, 0.0

            # Apply calibration to the INVERTED median AI depth inside the bounding box ROI
            ai_roi_patch = ai_depth_resized[py_min:py_max, px_min:px_max]
            ai_raw_median = float(np.median(ai_roi_patch))
            ai_inverted_median = 1.0 / (ai_raw_median + 1e-6)

            depth = float(s * ai_inverted_median + t)
            rospy.loginfo("Fused AI depth calculated successfully.")

            # -------------------------------------------------------------
            # TEMPORARY COMPARISON WITH OLD METRIC SERVER
            # -------------------------------------------------------------
            try:
                rospy.loginfo("Fetching uncalibrated metric model for comparison...")
                metric_depth_map = get_depth_data(found_encoded_image, url=METRIC_DEPTH_SERVER_URL)
                metric_depth_resized = cv2.resize(metric_depth_map, (hw_w, hw_h), interpolation=cv2.INTER_NEAREST)

                metric_roi_patch = metric_depth_resized[py_min:py_max, px_min:px_max]
                metric_median = float(np.median(metric_roi_patch))

                rospy.loginfo("\n================ DEPTH COMPARISON ================")
                rospy.loginfo("1. Calibrated Non-Metric (RANSAC): {:.3f} meters".format(depth))
                rospy.loginfo("2. Uncalibrated Metric (Raw AI)  : {:.3f} meters".format(metric_median))
                rospy.loginfo("Absolute Difference              : {:.3f} meters".format(abs(depth - metric_median)))
                rospy.loginfo("==================================================\n")
            except Exception as e:
                rospy.logerr("Failed to fetch from metric server for comparison: %s", str(e))
            # -------------------------------------------------------------

        rospy.loginfo("Final calculated depth: {:.2f} meters".format(depth))

        # Wait for spinning to finish
        self.spin_client.wait_for_result()
        rospy.sleep(0.5)

        self.navigate_client.send_goal_and_wait(ControllerNavigateGoal(
            target_u=center_u,
            target_v=center_v,
            depth=depth,
            image_id=found_image_id
        ))

        return True

    def spin_feedback_callback(self, feedback):
        rospy.loginfo("Pilot received picture %d!", feedback.image_id + 1)
        self.spin_image_queue.put((feedback.image_id, feedback.image_data, feedback.depth_data))

bridge = cv_bridge.CvBridge()

def encode_image(img):
    try:
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

    if hasattr(buffer, 'tobytes'):
        raw_bytes = buffer.tobytes()
    else:
        raw_bytes = buffer.tostring()

    return raw_bytes

def prompt_model(text, image):
    content = [{"type": "text", "text": text}]

    if image is not None:
        base64_image = base64.b64encode(image)
        if isinstance(base64_image, bytes):
            base64_image = base64_image.decode("utf-8")

        content.append({
            "type": "image_url",
            "image_url": {
                "url": "data:image/jpeg;base64,%s" % base64_image
            }
        })

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
        "Authorization": "Bearer EMPTY"
    }

    json_data = json.dumps(payload).encode("utf-8")
    request = urllib_req.Request(VLM_API_URL, data=json_data, headers=headers)

    try:
        response = urllib_req.urlopen(request)
        response_data = response.read()

        if isinstance(response_data, bytes):
            response_data = response_data.decode("utf-8")

        result = json.loads(response_data)
        return result["choices"][0]["message"]["content"]

    except Exception as e:
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
        rospy.logerr("Failed to parse JSON from LLM. Raw output was: %s", reply)

    return None

def detect_targets_in_image(instructions, image):
    reply = prompt_model(prompt_object_detection.format(instructions["target"], instructions["desc"]), image)
    try:
        items = []
        for item in json.loads(reply.strip()):
            box = item["box"]
            items.append({
                "desc": item["desc"],
                "box":  box,
            })
        return items

    except ValueError as e:
        rospy.logerr("Failed to parse JSON from LLM. Raw output was: %s", reply)
        return None

def get_depth_data(image, url=DEPTH_SERVER_URL):
    req = urllib_req.Request(url, data=image)
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
    pilot.execute_task("Go to the boy boxes on the counter!")

    rospy.spin()
