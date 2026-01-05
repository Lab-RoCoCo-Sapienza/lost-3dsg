#!/usr/bin/env python3

# Sopprimi warning
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
import os
import json
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TensorFlow warnings
import rclpy
from rclpy.node import Node
from rclpy.duration import Duration
from rclpy.qos import QoSProfile, DurabilityPolicy, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import Image, JointState, CameraInfo
from std_msgs.msg import String, Header
from visualization_msgs.msg import Marker, MarkerArray
from std_msgs.msg import ColorRGBA
from datetime import datetime
from sensor_msgs.msg import PointCloud2
import tf2_ros
from cv_bridge import CvBridge
import numpy as np
import cv2
from cv_utils import _clear_markers
import time
from dataclasses import dataclass
from typing import Tuple
from matplotlib.colors import to_rgb
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging
from datetime import datetime
import threading

# Project imports
from cv_utils import *
import utils
from utils import draw_detections, apply_nms
from models import DINO, VitSam, OWLv2
from tiago_project.msg import Centroid, CentroidArray, Bbox3d, Bbox3dArray
from object_info import Object
from world_model import wm
from tiago_project.msg import ObjectDescription, ObjectDescriptionArray

import torch
if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True

# Setup file logger and project paths
file_path = os.path.abspath(__file__)
# Find project root: if in install dir, go to workspace root, else go up to find CMakeLists.txt
current_dir = os.path.dirname(file_path)
PROJECT_ROOT = current_dir.split('/install/')[0] if '/install/' in current_dir else os.path.abspath(os.path.join(current_dir, "../.."))
log_dir = os.path.join(PROJECT_ROOT, "output")
os.makedirs(log_dir, exist_ok=True)

# Create log file with timestamp
log_file = os.path.join(log_dir, f"perception_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
file_handler = logging.FileHandler(log_file)
file_handler.setLevel(logging.DEBUG)
file_handler.setFormatter(logging.Formatter('%(asctime)s [%(levelname)s] %(message)s'))

# Logger for the perception module
module_logger = logging.getLogger('perception_module')
module_logger.setLevel(logging.DEBUG)
module_logger.addHandler(file_handler)

@dataclass
class Detection:
    bbox: Tuple[float, float, float, float] 
    label: str
    score: float
    mask: np.ndarray


class DetectObjects(Node):
    def __init__(self):
        super().__init__('detection_node')
        
        # Setup file logging for this node
        self.file_logger = module_logger
        self.file_logger.info("=== DetectObjects Node Initialized ===")
        self.get_logger().info(f"Log saved in: {log_file}")

        # CV Bridge for image conversions
        self.bridge = CvBridge()

        qos_latch = QoSProfile(depth=10, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        qos_standard = QoSProfile(depth=10)

        self.pub_image = self.create_publisher(Image, "/image_with_bb", qos_latch)
        self.COLORS = ['red', 'green', 'blue', 'magenta', 'gray', 'yellow'] * 3
        self.detector = OWLv2()
        self.vitsam = VitSam(utils.ENCODER_VITSAM_PATH, utils.DECODER_VITSAM_PATH)

        self.centroid_pub = self.create_publisher(CentroidArray, "/centroids_custom", qos_latch)

        self.tf_buffer = tf2_ros.Buffer(cache_time=Duration(seconds=30.0))
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        
        self.bbox_pub = self.create_publisher(Bbox3dArray, '/bbox_3d', qos_standard)
        self.pub_crop = self.create_publisher(Image, "/cropped_image", qos_standard)

        # Manage robot movement state
        self.is_stationary = True  
        self.time_stationary_start = None  
        self.last_detection_time = None 
        self.first_detection_done = False  
        self.min_stationary_after_movement = 2.0  

        # USED DURING TEST PHASE
        self.manual_trigger_requested = False
        self.waiting_for_input = False  
        
        self.create_subscription(JointState, "joint_states", self.joint_callback, qos_standard)
        self.head_joints = ['head_1_joint', 'head_2_joint']
        self.base_joints = ['wheel_left_joint', 'wheel_right_joint']
        self.position_threshold = 0.0015  # Threshold for position delta (rad)
        self.last_joint_positions = {}  # Previous positions to calculate delta


        # Synced camera data handler (recieve rgb, depth, camera_info, transform together)
        self.camera_data = utils.SyncedCameraData(self, sync_tolerance_ms=150)

        self.object_list=[]
        self.pub_object_descriptions = self.create_publisher(ObjectDescriptionArray, '/object_descriptions', qos_standard)
        
        self.bbox_marker_pub, self.centroid_marker_pub = init_bbox_publisher(self)

        self.pcl_objects_pub = self.create_publisher(PointCloud2, '/pcl_objects', qos_latch)
        self.pcl_objects_labels_pub = self.create_publisher(String, '/pcl_objects_labels', qos_latch)
        self.publish_individual_objects = False  # Flag to enable individual pcl publishing
        
        self.accumulated_points = [] 
        self.accumulated_labels = {} 
        self.pcl_object_id_counter = 0  
        
        self.individual_pcl_publishers = {} 

        # Flag to interrupt processing if the robot moves
        self.processing_interrupted = False

        self.filtered_objects = []

        # Clear accumulated data service
        self.clear_accumulated_markers()
        self.clear_accumulated_pointclouds()

    def log_both(self, level, message):
        # Log on ROS
        if level == 'info':
            self.get_logger().info(message)
        elif level == 'warn':
            self.get_logger().warn(message)
        elif level == 'error':
            self.get_logger().error(message)
        elif level == 'debug':
            self.get_logger().debug(message)
        
        # Log on file
        if level == 'info':
            self.file_logger.info(message)
        elif level == 'warn':
            self.file_logger.warning(message)
        elif level == 'error':
            self.file_logger.error(message)
        elif level == 'debug':
            self.file_logger.debug(message)

    def clear_accumulated_markers(self):
        """Clear all accumulated markers (centroids and bbox) and reset counters."""
        # Clear centroid markers
        try:
            _clear_markers("/centroid_markers", node=self, publisher=self.centroid_marker_pub)
            self.get_logger().info("Cleared accumulated centroid markers")
        except Exception as e:
            self.get_logger().warn(f"Error clearing centroids: {e}")

        # Clear bbox markers
        try:
            _clear_markers("/bbox_marker", node=self, publisher=self.bbox_marker_pub)
            self.get_logger().info("Cleared accumulated bbox markers")
        except Exception as e:
            self.get_logger().warn(f"Error clearing bbox: {e}")

        # Resetta i contatori
        if hasattr(self, '_centroid_marker_id_counter'):
            self._centroid_marker_id_counter = 0
        if hasattr(self, '_bbox_marker_id_counter'):
            self._bbox_marker_id_counter = 0

        self.get_logger().info("Contatori marker resettati")
    
    def clear_accumulated_pointclouds(self):
        """Cancella tutte le point clouds accumulate."""
        self.accumulated_points.clear()
        self.accumulated_labels.clear()
        self.pcl_object_id_counter = 0
        empty_cloud = PointCloud2()
        empty_cloud.header = Header(stamp=self.get_clock().now().to_msg(), frame_id="map")
        empty_cloud.height = 1
        empty_cloud.width = 0
        self.pcl_objects_pub.publish(empty_cloud)
        self.get_logger().info("Removed all the accumulated point clouds and reset counters.")
   
    def color_pcl(self, detections, data):
        """Generate and publish colored PointCloud2 from masks"""
        camera_info = data['camera_info']
        depth_img = data['depth']
        
        mask_list = []
        for detection in detections:
            mask = detection.mask
            mask2d = mask[:, :, 0]
            mask2d = (mask2d.astype(np.uint8) * 255)
            mask_list.append(mask2d)
        
        self.get_logger().info(f"color_pcl: processing {len(mask_list)} masks for PointCloud2")
        self.file_logger.info(f"color_pcl: processing {len(mask_list)} masks for PointCloud2")
        
        # Publish aggregated PointCloud2 for all objects
        try:
            labels = [det.label if hasattr(det, 'label') else f"obj_{i}" for i, det in enumerate(detections)]
            
            self.get_logger().info(f"Calling mask_list_to_pointcloud2 with {len(labels)} objects: {labels}")
            self.file_logger.info(f"Calling mask_list_to_pointcloud2 with {len(labels)} objects: {labels}")

            mask_list_to_pointcloud2(
                mask_list,
                depth_img,
                camera_info,
                node=self,
                labels=labels,
                frame_id=camera_info.header.frame_id, # Use camera frame 
                topic="/pcl_objects",
                max_points_per_obj=1500,
                publisher=self.pcl_objects_pub,
                labels_publisher=self.pcl_objects_labels_pub,
            )
            
            self.get_logger().info("mask_list_to_pointcloud2 completed")

            # Publish individual PointCloud2 for each object automatically if the user chooses
            try:
                # Set the flag to True if you wanna see individual pointclouds for each object
                if self.publish_individual_objects:
                    published_count = publish_individual_pointclouds_by_id(
                        mask_list,
                        depth_img,
                        camera_info,
                        node=self,
                        labels=labels,
                        frame_id="map",  # Use map for static room coordinates
                        topic_prefix="/pcl_id",
                        publishers_dict=self.individual_pcl_publishers,
                        id_counter_start=self.pcl_object_id_counter,
                        timestamp=camera_info.header.stamp,  # Use original image timestamp
                    )
                    # Update the counter for the next detection
                    self.pcl_object_id_counter += published_count
                    self.get_logger().info(f"publish_individual_pointclouds_by_id completed - {published_count} clouds published")
            except Exception as e:
                self.get_logger().error(f"Error in publish_individual_pointclouds_by_id: {e}")

        except Exception as e:
            self.get_logger().error(f"Errore publishing aggregated PointCloud2: {e}")
        
        return
    
    def run_detection(self, camera_data):
        self.get_logger().debug("Running detection...")
        
        self.log_both("info", "=== START DETECTION ===")
        self.log_both("info", "Calling VLM for object identification...")
        objects_to_identify = vlm_call(
            open(os.path.join(os.path.dirname(file_path), "object_identification_prompt.txt"), "r").read(),
            numpy_to_base64(camera_data['rgb'])
        )

        labels = []
        for label in objects_to_identify.split(','):
            labels.append(label.strip())

        self.log_both("info", f"Parsed labels ({len(labels)} objects): {labels}")

        self.detector.set_classes(labels)

        #inference 
        bboxs, labels, scores = self.detector.predict(camera_data['rgb'], box_threshold=0.32, text_threshold=0.25)
        
        
        self.log_both("info", f"OWLv2 detected {len(labels)} objects with labels: {labels}")

        # Apply NMS to remove overlapping bounding boxes
        if len(bboxs) > 0:
            bboxs, labels, scores = apply_nms(bboxs, labels, scores, iou_threshold=0.5)
            self.log_both("info", f"After NMS: {len(labels)} objects remaining with labels: {labels}")

        # Check movement after predict
        if not self.is_stationary:
            # ROS2_MIGRATION
            self.log_both("warn", "Robot moving after OWLv2 predict, interrupting")
            self.processing_interrupted = True
            return []

        #Obtain masks for each bbox
        
        detections = []
        for bbox, label_name, score in zip(bboxs, labels, scores):

            masks, _ = self.vitsam(camera_data['rgb'].copy(), bbox)
            for mask in masks:
                mask_np = mask.cpu().numpy() if hasattr(mask, 'cpu') else np.array(mask)
                mask_np = mask_np.transpose(1, 2, 0)
                detection = Detection(
                    bbox=tuple(bbox),
                    label=label_name,
                    score=score,
                    mask=mask_np
                )
                detections.append(detection)

        self.color_pcl(detections, camera_data)

        self.log_both("info", f"Finished detection with {len(detections)} final detections.")

        return detections

    def process_crop_vlm(self, crop_info):
        """Process a single cropped object from the original image using VLM."""
        if crop_info is None:
            return None
            
        label = crop_info['label']
        cropped = crop_info['cropped']
        # Load prompt from file
        prompt_file_path = os.path.join(os.path.dirname(file_path), "visual_prompt.txt")
        with open(prompt_file_path, "r") as f:
            visual_prompt_template = f.read().strip()
        
        visual_prompt = visual_prompt_template.replace("{LABEL}", label)
        encoded_image = numpy_to_base64(cropped)
        
        try:
            json_answer = vlm_call(visual_prompt, encoded_image)
            
            # Parsing JSON
            json_data = json.loads(json_answer)
            obj_data = json_data.get("objects", [{}])[0]
            
            return {
                "label": label,
                "json_answer": json_answer,
                "description": obj_data.get("description", "unknown"),
                "color": obj_data.get("color", "unknown"),
                "material": obj_data.get("material", "unknown"),
                "shape": obj_data.get("shape", "unknown")
            }
        except Exception as e:
            self.get_logger().error(f"Errore VLM per {label}: {e}")
            return {
                "label": label,
                "json_answer": "{}",
                "description": "unknown",
                "color": "unknown",
                "material": "unknown",
                "shape": "unknown"
            }


    def publish_objects(self):
        self.processing_interrupted = False
        self.log_both("info", "=== New perception cycle started ===")

        if not self.is_stationary:
            # ROS2_MIGRATION
            self.get_logger().warn("Robot in movimento all'inizio, annullo elaborazione")
            return

        # --- 1. Get synced data ---
        camera_data = self.camera_data.get_synced_data()

        if camera_data is None:
            self.get_logger().warn("Could not get synced camera data, waiting ...")
            return

        image_raw = camera_data['rgb']
        depth = camera_data['depth']
        camera_info = camera_data['camera_info']
        camera_frame = camera_data['camera_frame']

        # Save current transforms for cv utils script use
        self.current_transforms = {}
        self.current_transforms[(camera_frame, "map")] = camera_data['transform']

        image_crop_source = image_raw.copy()
        image_for_bb = image_raw.copy()

        # --- 2. Run detection (call object detector and publish point cloud for detected objects) ---
        detections = self.run_detection(camera_data)

        if self.processing_interrupted or not self.is_stationary:
            self.get_logger().error("Processing interrupted: robot moving during detection")
            return

        # Check if there are valid detections
        if not detections:
            # ROS2_MIGRATION
            self.get_logger().warn("No objects perceived")

            # IMPORTANT: Publish EMPTY ObjectDescriptionArray to tell the object_manager:
            # "I looked at the scene but found no valid objects"
            # This allows the object_manager to remove objects in the POV

            empty_cloud = PointCloud2()
            empty_cloud.header = Header(stamp=self.get_clock().now().to_msg(), frame_id=camera_info.header.frame_id)
            empty_cloud.height = 1
            empty_cloud.width = 0
            self.pcl_objects_pub.publish(empty_cloud)
            self.get_logger().info("Published empty PointCloud2 (0 points)")

            empty_descriptions = ObjectDescriptionArray()
            empty_descriptions.header = Header(stamp=self.get_clock().now().to_msg(), frame_id="map")
            self.pub_object_descriptions.publish(empty_descriptions)
            self.get_logger().warn("Published  ObjectDescriptionArray EMPTY (0 objects) - the object_manager can now remove objects in the POV")

            empty_bboxes = Bbox3dArray()
            empty_bboxes.header = Header(stamp=self.get_clock().now().to_msg(), frame_id="map")
            self.bbox_pub.publish(empty_bboxes)
            self.get_logger().info("Published empty Bbox3dArray (0 bboxes)")

            # Waiting for user input to start next perception cycle or automatic after movement
            self.waiting_for_input = False 
            if self.waiting_for_input:
                self.get_logger().info("==================================================")
                self.get_logger().info("Press ENTER to start the next perception cycle...")
                self.get_logger().info("==================================================")
                self.file_logger.info("Waiting for user input to start the next perception cycle")
 

            return

        img_with_bb = draw_detections(image_for_bb, detections)
                
        img_msg = self.bridge.cv2_to_imgmsg(img_with_bb, "bgr8")

        img_msg.header.stamp = self.get_clock().now().to_msg()
        img_msg.header.frame_id = camera_info.header.frame_id 


        self.pub_image.publish(img_msg)
        self.get_logger().info("Published final image_with_bb with BBoxes")

        # --- Save visualizations in output folder DEBUG STEP ---
        viz_output_dir = os.path.join(PROJECT_ROOT, "output/visualizations")
        os.makedirs(viz_output_dir, exist_ok=True)
        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        bbox_img_path = os.path.join(viz_output_dir, f"bbox_{timestamp_str}.jpg")
        cv2.imwrite(bbox_img_path, img_with_bb)
        self.get_logger().info(f"Saved image with bbox: {bbox_img_path}")

        # Save depth visualization with colored masks
        depth_viz_dir = os.path.join(viz_output_dir, "depth")
        os.makedirs(depth_viz_dir, exist_ok=True)
        # Normalize depth for visualization
        depth_normalized = cv2.normalize(depth, None, 0, 255, cv2.NORM_MINMAX)
        depth_colored = cv2.applyColorMap(depth_normalized.astype(np.uint8), cv2.COLORMAP_JET)
        depth_viz_path = os.path.join(depth_viz_dir, f"depth_{timestamp_str}.jpg")
        cv2.imwrite(depth_viz_path, depth_colored)
        self.get_logger().info(f"Saved depth visualization: {depth_viz_path}")

        # Save projected point cloud visualization
        pcl_viz_dir = os.path.join(viz_output_dir, "pointclouds")
        os.makedirs(pcl_viz_dir, exist_ok=True)

        # Create image with all colored masks overlaid
        pcl_overlay = image_raw.copy()
        colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255), (0, 255, 255)]
        for idx, det in enumerate(detections):
            mask_2d = (det.mask[:, :, 0] * 255).astype(np.uint8)
            color_bgr = colors[idx % len(colors)]
            color_rgb = (color_bgr[2]/255.0, color_bgr[1]/255.0, color_bgr[0]/255.0)
            overlay_mask_on_image(pcl_overlay, mask_2d, color_rgb=color_rgb, alpha=0.4)

            # Add label text at centroid
            moments = cv2.moments(mask_2d)
            if moments['m00'] != 0:
                cx = int(moments['m10'] / moments['m00'])
                cy = int(moments['m01'] / moments['m00'])
                cv2.putText(pcl_overlay, det.label, (cx, cy),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, color_bgr, 2)

        pcl_overlay_path = os.path.join(pcl_viz_dir, f"pcl_overlay_{timestamp_str}.jpg")
        cv2.imwrite(pcl_overlay_path, pcl_overlay)
        self.get_logger().info(f"Saved PCL overlay visualization: {pcl_overlay_path}")
        ## FINISH DEBUG STEP ###
        
        # --- 3. Prepare for VLM calls for each detected object in parallel ---
        centroids_3d = []
        bboxes_3d = []
        labels1 = []
        descriptions = []  
        for idx, det in enumerate(detections):                
            labels1.append(det.label)

        # BBOX3D message
        msg = Bbox3dArray()
        msg.header = Header(stamp=self.get_clock().now().to_msg(), frame_id="map")

        # --- 4. Calculate 3D centroids and bboxes for ALL detections together ---
        all_masks = [det.mask[:, :, 0] for det in detections]
        centroids_3d, bboxes_3d = mask_list_to_centroid_and_bbox(
            all_masks, labels1, depth, camera_info, node=self, bbox_marker_pub=self.bbox_marker_pub, centroid_marker_pub=self.centroid_marker_pub
        )

        # --- 5. Prepare crop images ---
        crops_data = []
        h, w = image_crop_source.shape[:2]
        
        for idx, det in enumerate(detections):   
            bbox_3d = bboxes_3d[idx]

            # --- CROP object image ---

            x_min, y_min, x_max, y_max = map(int, det.bbox)

            # Validate bbox coordinates
            x_min = max(0, min(x_min, w - 1))
            y_min = max(0, min(y_min, h - 1))
            x_max = max(x_min + 1, min(x_max, w))
            y_max = max(y_min + 1, min(y_max, h))

            cropped = image_crop_source[y_min:y_max, x_min:x_max].copy()

            # Check for invalid crops
            if cropped.size == 0 or cropped.shape[0] == 0 or cropped.shape[1] == 0:
                self.get_logger().warn(f"Invalid crop for {det.label}: bbox={det.bbox}, image dimensions={h}x{w}")
                crops_data.append(None)
                continue

            crops_data.append({
                'cropped': cropped,
                'label': det.label,
                'idx': idx
            })

        # --- Publish all crops ---
        for crop_info in crops_data:
            if crop_info is None:
                continue
            try:
                crop_msg = self.bridge.cv2_to_imgmsg(crop_info['cropped'], encoding="bgr8")
                crop_msg.header.stamp = self.get_clock().now().to_msg()
                crop_msg.header.frame_id = "camera"
                self.pub_crop.publish(crop_msg)
                self.get_logger().info(f"Pubblicato crop per {crop_info['label']}")
            except Exception as e:
                self.get_logger().error(f"Errore nella conversione del crop: {e}")

        # --- 6. Call VLM for each cropped object in parallel ---
        with ThreadPoolExecutor(max_workers=min(4, len(crops_data))) as executor:
            # Submit all tasks
            future_to_crop = {executor.submit(self.process_crop_vlm, crop_info): crop_info 
                             for crop_info in crops_data}
            
            # Collect results as they complete
            vlm_results = [None] * len(crops_data)
            for future in as_completed(future_to_crop):
                result = future.result()
                if result:
                    for i, crop_info in enumerate(crops_data):
                        if crop_info and crop_info['label'] == result['label']:
                            vlm_results[i] = result
                            self.get_logger().info(f"Completed VLM call for {result['label']}")
                            break

        # --- Save JSON and prepare descriptions for DEBUG ---
        output_dir = os.path.expanduser(os.path.join(os.path.dirname(file_path), "../../output"))
        os.makedirs(output_dir, exist_ok=True)
        
        # Iterate over DETECTIONS to match result, bbox, and centroid
        for idx, det in enumerate(detections):
            result = vlm_results[idx] if idx < len(vlm_results) else None
            bbox_3d = bboxes_3d[idx] if idx < len(bboxes_3d) else None
            
            if result is None:
                descriptions.append({
                    "description": "unknown",
                    "color": "unknown",
                    "material": "unknown",
                    "shape": "unknown"
                })
            else:                
                descriptions.append({
                    "description": result['description'],
                    "color": result['color'],
                    "material": result['material'],
                    "shape": result['shape']
                }) 
            
            if bbox_3d is not None:
                box_msg = Bbox3d()
                box_msg.label = det.label
                box_msg.x_min = bbox_3d["x_min"]
                box_msg.y_min = bbox_3d["y_min"]
                box_msg.z_min = bbox_3d["z_min"]
                box_msg.x_max = bbox_3d["x_max"]
                box_msg.y_max = bbox_3d["y_max"]
                box_msg.z_max = bbox_3d["z_max"]
                msg.boxes.append(box_msg)

        self.bbox_pub.publish(msg)

        # --- 6. Publish ObjectDescriptionArray ---
        object_desc_array_msg = ObjectDescriptionArray()
        object_desc_array_msg.header = Header(stamp=self.get_clock().now().to_msg(), frame_id="map")

        for det, desc_dict in zip(detections, descriptions):
            o = ObjectDescription()
            o.label = det.label
            o.description = desc_dict["description"]
            o.color = desc_dict["color"]
            o.material = desc_dict["material"]
            o.shape = desc_dict["shape"]
            object_desc_array_msg.descriptions.append(o)

        self.pub_object_descriptions.publish(object_desc_array_msg)
    
        wm.actual_perceptions.clear()
        self.clear_accumulated_markers()

        # --- Update World Model with actual perceptions ---
        for det, centroid, bbox, desc_dict in zip(detections, centroids_3d, bboxes_3d, descriptions):
            wm.add_actual_perception(Object(
                det.label, 
                centroid, 
                bbox, 
                description=desc_dict["description"],
                color=desc_dict["color"],
                material=desc_dict["material"],
                shape=desc_dict["shape"]
            ))

        perceptions_output_dir = os.path.join(PROJECT_ROOT, "output")
        os.makedirs(perceptions_output_dir, exist_ok=True)
        perceptions_output_path = os.path.join(perceptions_output_dir, "actual_perceptions.json")

        ## FOR DEBUG: save actual perceptions to JSON
        new_perceptions = [
            {
                "label": obj.label,
                "centroid": obj.centroid.tolist() if hasattr(obj.centroid, 'tolist') else (list(obj.centroid) if obj.centroid is not None else None),
                "bbox": obj.bbox,
                "description": obj.description,
                "color": obj.color,
                "material": obj.material,
                "shape": obj.shape
            }
            for obj in wm.actual_perceptions
        ]

        with open(perceptions_output_path, "a") as f:
            json.dump(new_perceptions, f, indent=4)

        # Wait for user input to start next perception cycle or automatic after movement
        self.waiting_for_input = False

        self.log_both('info', "=== PUBLISH_OBJECTS COMPLETED - Exiting function ===")
        if self.waiting_for_input:
            self.get_logger().info("==================================================")
            self.get_logger().info("Press ENTER to start the next perception cycle...")
            self.get_logger().info("==================================================")
            self.file_logger.info("Waiting for user input to start the next perception cycle")

    def joint_callback(self, msg):
        """Callback to monitor robot joint states and detect movement."""
        deltas = []

        for j in self.head_joints + self.base_joints:
            if j in msg.name:
                idx = msg.name.index(j)
                pos = msg.position[idx]

                # Calcola delta posizione
                if j in self.last_joint_positions:
                    delta = abs(pos - self.last_joint_positions[j])
                    deltas.append(delta)

                self.last_joint_positions[j] = pos

        if not deltas:
            return

        # Compute mean delta position
        media_delta = sum(deltas) / len(deltas)
        was_stationary = self.is_stationary
        currently_stationary = media_delta < self.position_threshold

        if currently_stationary and not was_stationary:
            # Robot just stopped after movement - do NOT reset timer here
            # Timer will be reset after detection completes
            self.is_stationary = True
            msg = f"Robot stopped. Timer will trigger next detection after {self.min_stationary_after_movement}s"
            self.log_both('info', msg)
        elif currently_stationary and was_stationary:
            # Already stationary - nothing to do, timer continues from last detection
            self.is_stationary = True
        elif not currently_stationary:
            # Robot is moving - stop any ongoing processing
            self.is_stationary = False
            if was_stationary:
                self.log_both('warn', "MOVEMENT DETECTED! The robot has moved.")
                self.file_logger.warning(f"  Average joint delta: {media_delta:.8f} rad")
                self.processing_interrupted = True


def main(args=None):
    rclpy.init(args=args)
    node = DetectObjects()

    # Thread to handle terminal input (for test it can be useful to trigger manually the perception)
    def input_thread():
        while rclpy.ok():
            try:
                # Wait for user input (blocking)
                input()  # Wait for ENTER

                # When ENTER is received, set the flag
                if node.waiting_for_input:
                    node.manual_trigger_requested = True
                    node.waiting_for_input = False
                    node.get_logger().info("Input received! Starting perception...")
                elif not node.first_detection_done:
                    # Allow triggering even before the first detection
                    node.manual_trigger_requested = True
                    node.get_logger().info("Input received! Starting first perception...")
            except:
                break

    # Start thread for input (ENABLE ONLY DURING TEST and set waiting_for_input = True where needed)
    #input_handler = threading.Thread(target=input_thread, daemon=True)
    #input_handler.start()

    def timer_callback():
        current_time = node.get_clock().now()

        # Caso TRIGGER MANUALE - ha priorità su tutto
        if node.manual_trigger_requested:
            if node.camera_data.get_synced_data() is not None:
                node.log_both('info', "=== MANUAL PERCEPTION TRIGGERED ===")
                node.publish_objects()
                # Capture time AFTER publish_objects completes
                completion_time = node.get_clock().now()
                node.first_detection_done = True
                node.last_detection_time = completion_time
                node.time_stationary_start = completion_time  # Reset timer after detection
                node.manual_trigger_requested = False  # Reset flag
            else:
                node.get_logger().warn("Trigger ricevuto ma dati camera non disponibili - riprovo al prossimo ciclo")
            return

        # Case 1: FIRST DETECTION
        if not node.first_detection_done:
            if node.camera_data.get_synced_data() is not None:
                msg = "First detection: synchronized data available – starting perception."
                node.log_both('info', msg)
                node.publish_objects()
                # Capture time AFTER publish_objects completes
                completion_time = node.get_clock().now()
                node.first_detection_done = True
                node.last_detection_time = completion_time
                node.time_stationary_start = completion_time  # Reset timer after first detection
            else:
                node.file_logger.debug("Attendo dati sincronizzati per prima detection...")
            return

        # Case 2: RE-DETECTION AFTER TIME INTERVAL (Set the time interval as you prefer) now for two seconds
        print("________________ Timer callback: checking for re-detection conditions... __________________")
        print("Stationary:", node.is_stationary)
        print("Time stationary for : ", (current_time - node.time_stationary_start).nanoseconds / 1e9 if node.time_stationary_start else "N/A")
        print("Threshold: ", node.min_stationary_after_movement)

        if node.is_stationary and node.time_stationary_start is not None:
            time_stationary = (current_time - node.time_stationary_start).nanoseconds / 1e9
            print(f"DEBUG: time_stationary={time_stationary:.2f}s >= threshold={node.min_stationary_after_movement}s? {time_stationary >= node.min_stationary_after_movement}")

            if time_stationary >= node.min_stationary_after_movement:
                # Robot is stationary for enough time, data are enough stable
                node.log_both('info', "Robot stationary for sufficient time – starting new perception cycle.")
                print("DEBUG: About to call publish_objects()")
                node.publish_objects()
                print("DEBUG: publish_objects() completed, resetting timer")
                # Capture time AFTER publish_objects completes
                completion_time = node.get_clock().now()
                node.last_detection_time = completion_time
                node.time_stationary_start = completion_time  # Reset timer after detection
                print(f"DEBUG: Timer reset to completion_time (not current_time from start of callback)")
            else:
                print(f"DEBUG: Not enough time yet, waiting {node.min_stationary_after_movement - time_stationary:.2f}s more")
            return
        else:
            print(f"DEBUG: Skipping re-detection - is_stationary={node.is_stationary}, time_stationary_start={'None' if node.time_stationary_start is None else 'set'}")

    _timer = node.create_timer(0.5, timer_callback)  

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()