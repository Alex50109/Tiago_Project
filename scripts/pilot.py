#!/usr/bin/env python

import Queue as queue
import base64
import json
import cv2
import rospy
import actionlib
import cv_bridge
import urllib2 as urllib_req
import zlib
import numpy as np

from tiago_project.msg import ControllerFindObjectAction, ControllerFindObjectGoal
from tiago_project.msg import TargetUpdate
from tiago_project.srv import VoiceCommand, VoiceCommandResponse
from tiago_project.prompts import prompt_instruction_parser, prompt_object_detection

SERVER_IP = "10.41.3.112"

VLM_API_URL = "http://{}:8000/v1/chat/completions".format(SERVER_IP)

# Port 9000 for the Metric Depth Model
METRIC_DEPTH_SERVER_URL = "http://{}:9000/predict_depth_raw".format(SERVER_IP)

SPIN_STEP_ANGLE = 45.0
SPIN_STEPS = 8
VLM_IMAGE_SIZE = (640.0, 480.0)

TERMINAL_GOAL_STATES = [
    actionlib.GoalStatus.SUCCEEDED,
    actionlib.GoalStatus.ABORTED,
    actionlib.GoalStatus.PREEMPTED,
    actionlib.GoalStatus.REJECTED,
    actionlib.GoalStatus.LOST,
]

class Pilot:
    def __init__(self):
        self.find_client = actionlib.SimpleActionClient('find_object', ControllerFindObjectAction)
        self.update_pub = rospy.Publisher('/target_update', TargetUpdate, queue_size=1)

        rospy.loginfo("Pilot waking up... Waiting for controller to come online.")

        self.find_client.wait_for_server()

        rospy.loginfo("Controller linked. Ready to command!")

        self.image_queue = queue.Queue(SPIN_STEPS)

    def clear_image_queue(self):
        # clear queue (maybe there is a better way)
        while True:
            try:
                self.image_queue.get_nowait()
            except queue.Empty:
                break

    def execute_task(self, instructions):
        target = parse_instructions_prompt(instructions)
        if target is None:
            rospy.logerr("Not an interesting command! I have the right to ignore you!")
            return False

        rospy.loginfo("Command is to go to %s (keep %s in mind)!", target["target"], target["desc"])

        self.clear_image_queue()
        self.find_client.send_goal(
            ControllerFindObjectGoal(num_pictures=SPIN_STEPS, step_angle=SPIN_STEP_ANGLE, navigate=True),
            feedback_cb=self.feedback_callback
        )

        update_sent = False

        while not rospy.is_shutdown():
            if self.find_client.get_state() in TERMINAL_GOAL_STATES:
                break

            try:
                image_id, image_data, depth_data = self.image_queue.get(timeout=1.0)
            except queue.Empty:
                continue

            encoded_image = encode_image(image_data)
            if encoded_image is None:
                continue

            rospy.loginfo("Prompting picture %d!", image_id + 1)
            detections = detect_targets_in_image(target, encoded_image)
            rospy.loginfo("LLM returned: %s", detections)

            if not detections:
                continue

            depth = compute_object_depth(depth_data, encoded_image, detections[0]["box"])
            if depth is None:
                rospy.logwarn("Depth estimation failed for picture %d. Trying the next frame.", image_id + 1)
                continue

            center_u, center_v = detection_center(detections[0]["box"])

            update = TargetUpdate()
            update.header.stamp = rospy.Time.now()
            update.image_id = image_id
            update.target_u = center_u
            update.target_v = center_v
            update.depth = depth
            self.update_pub.publish(update)

            if not update_sent:
                rospy.loginfo("Target found at {:.2f}m! Steering the navigation with live updates.".format(depth))
                update_sent = True
                self.clear_image_queue()

        result = self.find_client.get_result()
        if result is not None:
            rospy.loginfo("Mission over! target_found=%s reached=%s", result.target_found, result.reached)
            return result.reached

        return update_sent

    def feedback_callback(self, feedback):
        rospy.loginfo("Pilot received picture %d!", feedback.image_id + 1)
        if feedback.navigating:
            # only the freshest frame matters while driving
            self.clear_image_queue()
        self.image_queue.put((feedback.image_id, feedback.image_data, feedback.depth_data))

    def voice_command_service(self, request):
        """
        ROS service handler for /tiago/voice_command.

        Blocks until the task is finished, mirroring the synchronous
        `rosservice call` performed by the main client's ROSClient.
        """
        rospy.loginfo("Voice command received: %s", request.command)

        try:
            reached = self.execute_task(request.command)
        except Exception as e:
            rospy.logerr("Voice command failed: %s", str(e))
            return VoiceCommandResponse(success=False, message=str(e))

        if reached:
            return VoiceCommandResponse(success=True, message="Task completed: target reached.")
        return VoiceCommandResponse(success=False, message="Task finished, but the target was not reached.")

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

    # Encode as PNG instead of JPEG to preserve quality
    success, buffer = cv2.imencode('.png', cv_img)

    if not success:
        rospy.logerr("Failed to encode image to PNG")
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
                "url": "data:image/png;base64,%s" % base64_image
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
        response = urllib_req.urlopen(request, timeout=30)
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

def normalize_box(box):
    """Clips a VLM bounding box from the 0-1000 scale to normalized [0, 1] coordinates."""
    x_min, y_min, x_max, y_max = box
    u_min = min(max(x_min / 1000.0, 0.0), 1.0)
    v_min = min(max(y_min / 1000.0, 0.0), 1.0)
    u_max = min(max(x_max / 1000.0, 0.0), 1.0)
    v_max = min(max(y_max / 1000.0, 0.0), 1.0)
    return u_min, v_min, u_max, v_max

def detection_center(box):
    """Normalized (u, v) center of a VLM bounding box."""
    u_min, v_min, u_max, v_max = normalize_box(box)
    return (u_min + u_max) / 2.0, (v_min + v_max) / 2.0

def compute_object_depth(depth_data, encoded_image, box):
    """
    Median depth of the object's central bounding box ROI, measured with the
    hardware depth camera or, when the hardware is blind, with the metric AI model.
    Returns None when no trustworthy depth can be produced.
    """
    try:
        hw_depth_raw = bridge.imgmsg_to_cv2(depth_data, desired_encoding="passthrough")
    except cv_bridge.CvBridgeError as e:
        rospy.logerr("CvBridge Error: %s" % str(e))
        return None

    # Convert 16UC1 (millimeters) to meters if necessary
    if hw_depth_raw.dtype == np.uint16:
        hw_depth = hw_depth_raw.astype(np.float32) / 1000.0
    else:
        hw_depth = hw_depth_raw.copy()

    hw_h, hw_w = hw_depth.shape

    u_min, v_min, u_max, v_max = normalize_box(box)

    # Extract Center 50% ROI of the bounding box (ROI = region of interest)
    roi_u_min = u_min + (u_max - u_min) * 0.25
    roi_u_max = u_max - (u_max - u_min) * 0.25
    roi_v_min = v_min + (v_max - v_min) * 0.25
    roi_v_max = v_max - (v_max - v_min) * 0.25

    px_min = max(0, int(roi_u_min * (hw_w - 1)))
    px_max = min(hw_w, max(px_min + 1, int(roi_u_max * (hw_w - 1))))
    py_min = max(0, int(roi_v_min * (hw_h - 1)))
    py_max = min(hw_h, max(py_min + 1, int(roi_v_max * (hw_h - 1))))

    # -------------------------------------------------------------
    # DEBUG IMAGE GENERATION: Draw Bounding Box and ROI
    # -------------------------------------------------------------
    try:
        # 1. Decode the saved PNG bytes back to a numpy BGR image
        debug_rgb = cv2.imdecode(np.fromstring(encoded_image, np.uint8), cv2.IMREAD_COLOR)

        # 2. Normalize and colorize the raw hardware depth map for human viewing
        # Clip depth to 5 meters for better contrast, then scale to 0-255
        depth_clipped = np.clip(hw_depth, 0, 5.0)
        depth_normalized = ((depth_clipped / 5.0) * 255.0).astype(np.uint8)
        debug_depth_color = cv2.applyColorMap(depth_normalized, cv2.COLORMAP_JET)

        # Calculate the full bounding box pixel coordinates
        bb_x_min = int(u_min * hw_w)
        bb_x_max = int(u_max * hw_w)
        bb_y_min = int(v_min * hw_h)
        bb_y_max = int(v_max * hw_h)

        # Draw the Full Bounding Box (Red, thick line)
        cv2.rectangle(debug_rgb, (bb_x_min, bb_y_min), (bb_x_max, bb_y_max), (0, 0, 255), 2)
        cv2.rectangle(debug_depth_color, (bb_x_min, bb_y_min), (bb_x_max, bb_y_max), (0, 0, 255), 2)

        # Draw the Center 50% ROI Box (Green, thick line)
        cv2.rectangle(debug_rgb, (px_min, py_min), (px_max, py_max), (0, 255, 0), 2)
        cv2.rectangle(debug_depth_color, (px_min, py_min), (px_max, py_max), (0, 255, 0), 2)

        # Save the images to disk
        cv2.imwrite("/tmp/debug_tiago_rgb.jpg", debug_rgb)
        cv2.imwrite("/tmp/debug_tiago_depth.jpg", debug_depth_color)
        rospy.loginfo("Saved debug images to /tmp/debug_tiago_rgb.jpg and /tmp/debug_tiago_depth.jpg")

    except Exception as e:
        rospy.logerr("Failed to generate debug images: %s", str(e))
    # -------------------------------------------------------------

    hw_roi_patch = hw_depth[py_min:py_max, px_min:px_max]

    # Clean NumPy approach: Get only valid hardware pixels FIRST to prevent any warnings
    roi_finite_mask = np.isfinite(hw_roi_patch)
    roi_finite_pixels = hw_roi_patch[roi_finite_mask]

    # Filter by distance on strictly valid numbers
    valid_hw_pixels = roi_finite_pixels[(roi_finite_pixels > 0.2) & (roi_finite_pixels < 3.0)]

    depth = None

    # Require at least 30% of the patch size to trust hardware
    if len(valid_hw_pixels) > (hw_roi_patch.size * 0.30):
        depth = float(np.median(valid_hw_pixels))
        rospy.loginfo("Hardware depth acquired successfully. Object is close.")
    else:
        rospy.loginfo("Hardware depth blind in bounding box. Using AI model...")
        try:
            rospy.loginfo("Fetching metric depth model data...")
            metric_depth_map = get_depth_data(encoded_image, url=METRIC_DEPTH_SERVER_URL)
            metric_depth_resized = cv2.resize(metric_depth_map, (hw_w, hw_h), interpolation=cv2.INTER_NEAREST)

            metric_roi_patch = metric_depth_resized[py_min:py_max, px_min:px_max]
            depth = float(np.median(metric_roi_patch))

        except Exception as e:
            rospy.logerr("Failed to fetch from metric depth server: %s", str(e))

    if depth is not None:
        rospy.loginfo("Final calculated depth: {:.2f} meters".format(depth))

    return depth

def get_depth_data(image, url=METRIC_DEPTH_SERVER_URL):
    req = urllib_req.Request(url, data=image)
    req.add_header('Content-Type', 'application/octet-stream')
    req.add_header('Content-Length', str(len(image)))

    response = urllib_req.urlopen(req, timeout=30)

    h = int(response.headers.get('X-Depth-Height'))
    w = int(response.headers.get('X-Depth-Width'))

    compressed_data = response.read()
    raw_bytes = zlib.decompress(compressed_data)

    depth_map = np.fromstring(raw_bytes, dtype=np.float32).reshape((h, w))
    return depth_map

if __name__ == '__main__':
    rospy.init_node('pilot_interface', anonymous=True)

    pilot = Pilot()
    voice_service = rospy.Service('/tiago/voice_command', VoiceCommand, pilot.voice_command_service)
    rospy.loginfo("Pilot ready! Voice command service available at /tiago/voice_command.")

    try:
        rospy.spin()
    except (rospy.ROSInterruptException, KeyboardInterrupt):
        pass
