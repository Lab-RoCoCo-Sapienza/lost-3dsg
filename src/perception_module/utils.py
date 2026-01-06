from object_info import Object
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from rclpy.time import Time as ROS2Time
from rclpy.duration import Duration as ROS2Duration
import numpy as np
from cv_bridge import CvBridge
import cv2, os, colorsys
from sensor_msgs.msg import Image, CameraInfo
from scipy.spatial import KDTree
from std_msgs.msg import ColorRGBA


bridge = CvBridge()

file_path = os.path.abspath(__file__)
ENCODER_VITSAM_PATH = os.path.join(os.path.dirname(file_path),"utils", "l2_encoder.onnx")
DECODER_VITSAM_PATH = os.path.join(os.path.dirname(file_path),"utils", "l2_decoder.onnx")


class SyncedCameraData:
    """
    Manages the reception of RGB, Depth, CameraInfo and Transform from the robot.
    ALWAYS updates with the most recent frames (no temporal synchronization).
    """
    def __init__(self, node, sync_tolerance_ms=50):
        """
        Args:
            node: ROS2 Node instance
            sync_tolerance_ms: Not used (kept for compatibility)
        """
        self.node = node
        self.bridge = CvBridge()

        # Data cache - ALWAYS UPDATED with the most recent messages
        self.cached_rgb = None
        self.cached_depth = None
        self.cached_camera_info = None
        self.cached_transform = None
        self.all_ready = False

        # QoS for real robot sensor topics
        qos_sensor = QoSProfile(
            depth=10,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST
        )

        # Persistent subscriptions
        self.node.get_logger().info("Subscribing to camera topics...")
        node.create_subscription(Image, '/head_front_camera/rgb/image_raw', self._rgb_callback, qos_sensor)
        node.create_subscription(Image, '/head_front_camera/depth/image_raw', self._depth_callback, qos_sensor)
        node.create_subscription(CameraInfo, '/head_front_camera/rgb/camera_info', self._camera_info_callback, qos_sensor)
        self.node.get_logger().info("Subscriptions created!")

    def _rgb_callback(self, msg):
        """ALWAYS updates with the most recent RGB"""
        first_time = self.cached_rgb is None
        self.cached_rgb = msg  # Always update!
        if first_time:
            self.node.get_logger().info("RGB received (first frame)")
        # Always try to get the transform
        self._try_get_transform()

    def _depth_callback(self, msg):
        """ALWAYS updates with the most recent Depth"""
        first_time = self.cached_depth is None
        self.cached_depth = msg  # Always update!
        if first_time:
            self.node.get_logger().info("Depth received (first frame)")
            self._check_all_ready()

    def _camera_info_callback(self, msg):
        """Saves CameraInfo (usually doesn't change)"""
        if self.cached_camera_info is None:
            self.cached_camera_info = msg
            self.node.get_logger().info("CameraInfo received")
            self._check_all_ready()

    def _try_get_transform(self):
        """ALWAYS updates the transform with the most recent one available"""
        if self.cached_rgb is None:
            return  # Don't have RGB yet

        if not hasattr(self.node, 'tf_buffer'):
            return

        try:
            camera_frame = self.cached_rgb.header.frame_id
            target_frame = "map"

            # ALWAYS use the most recent transform available (timestamp=0)
            # We ignore the image timestamp because there's too much delay
            transform = self.node.tf_buffer.lookup_transform(
                target_frame,
                camera_frame,
                ROS2Time(seconds=0),  # Most recent available
                timeout=ROS2Duration(seconds=0.01)
            )

            first_time = self.cached_transform is None
            self.cached_transform = transform  # Always update!

            if first_time:
                self.node.get_logger().info("✓ Transform received (first)")
                self._check_all_ready()

        except Exception as e:
            if not hasattr(self, '_transform_error_logged'):
                self.node.get_logger().warn(f"Transform not available: {e}")
                self._transform_error_logged = True

    def _check_all_ready(self):
        """Checks if we have ALL the data"""
        if (self.cached_rgb is not None and
            self.cached_depth is not None and
            self.cached_camera_info is not None and
            self.cached_transform is not None):
            if not self.all_ready:
                self.node.get_logger().info("OK - All data ready!")
                self.all_ready = True

    def get_synced_data(self):
        """
        Returns camera data ONLY if ALL are available (RGB, Depth, CameraInfo, Transform).

        Returns:
            dict or None: {
                'rgb': numpy array BGR,
                'depth': numpy array (meters),
                'camera_info': CameraInfo msg,
                'transform': TransformStamped,
                'timestamp': Time of RGB frame,
                'camera_frame': str
            }
        """
        # Check if we have ALL the data (including transform!)
        if (self.cached_rgb is None or
            self.cached_depth is None or
            self.cached_camera_info is None or
            self.cached_transform is None):
            return None

        try:
            # Convert images
            rgb_cv = self.bridge.imgmsg_to_cv2(self.cached_rgb, 'bgr8')
            depth_array = self.bridge.imgmsg_to_cv2(self.cached_depth, desired_encoding='passthrough')
            depth_array = np.asarray(depth_array).astype(float)

            # Convert to meters if necessary (depth from Asus Xtion is in mm)
            if depth_array.max() > 20.0:
                depth_array = depth_array / 1000.0

            return {
                'rgb': rgb_cv,
                'depth': depth_array,
                'camera_info': self.cached_camera_info,
                'transform': self.cached_transform,
                'timestamp': self.cached_rgb.header.stamp,
                'camera_frame': self.cached_rgb.header.frame_id
            }

        except Exception as e:
            self.node.get_logger().error(f"Data conversion error: {e}")
            return None


def depth_image_to_point_cloud(depth_image, camera_intrinsics):
    """Convert depth image to 3D point cloud"""
    height, width = depth_image.shape

    v, u = np.indices((height, width))

    x = (u - camera_intrinsics[0, 2]) * depth_image / camera_intrinsics[0, 0]
    y = (v - camera_intrinsics[1, 2]) * depth_image / camera_intrinsics[1, 1]
    z = depth_image

    points = np.dstack((x, y, z)).reshape(-1, 3)

    return points


def statistical_outlier_removal(points_xyz, k=20, std_ratio=2.0):
    """
    Removes statistical outliers based on the mean distance from the k nearest neighbors.

    Args:
        points_xyz: Numpy array (N, 3) with XYZ coordinates
        k: Number of neighbors to consider
        std_ratio: Standard deviation multiplier for the threshold

    Returns:
        mask: Boolean array (N,) where True = valid point
    """

    if len(points_xyz) < k:
        return np.ones(len(points_xyz), dtype=bool)

    tree = KDTree(points_xyz)
    distances, _ = tree.query(points_xyz, k=k+1)  # +1 because it includes the point itself
    mean_distances = distances[:, 1:].mean(axis=1)  # Exclude the point itself (distance 0)

    global_mean = mean_distances.mean()
    global_std = mean_distances.std()
    threshold = global_mean + std_ratio * global_std

    mask = mean_distances < threshold
    return mask

def get_distinct_color(index):
    """
    Generate a distinct color for each index using HSV.
    Returns a ColorRGBA.
    """
    hue = (index * 0.618033988749895) % 1.0  # Golden ratio for uniform distribution
    saturation = 0.8
    value = 0.9
    r, g, b = colorsys.hsv_to_rgb(hue, saturation, value)
    return ColorRGBA(r=r, g=g, b=b, a=1.0)

def apply_nms(bboxs, labels, scores, iou_threshold=0.5):
    """
    Apply CLASS-AWARE Non-Maximum Suppression to remove overlapping bounding boxes.
    NMS is applied SEPARATELY for each class, so boxes of different classes are never suppressed.
    This prevents removing a "book" just because it overlaps with a "table".

    Args:
        bboxs: List of bounding boxes [x1, y1, x2, y2]
        labels: List of corresponding labels
        scores: List of confidence scores
        iou_threshold: IoU threshold to consider two boxes as overlapping

    Returns:
        bboxs_filtered, labels_filtered, scores_filtered
    """
    if len(bboxs) == 0:
        return [], [], []

    # Convert to numpy arrays for easier manipulation
    bboxs = np.array(bboxs)
    scores = np.array(scores)
    labels = np.array(labels)

    # Apply NMS separately for each unique class
    unique_labels = np.unique(labels)
    all_keep_indices = []

    for label in unique_labels:
        # Get indices for this class only
        class_mask = labels == label
        class_indices = np.where(class_mask)[0]

        if len(class_indices) == 0:
            continue

        class_bboxs = bboxs[class_indices]
        class_scores = scores[class_indices]

        # Calculate areas
        x1 = class_bboxs[:, 0]
        y1 = class_bboxs[:, 1]
        x2 = class_bboxs[:, 2]
        y2 = class_bboxs[:, 3]
        areas = (x2 - x1) * (y2 - y1)

        # Sort by score (descending)
        order = class_scores.argsort()[::-1]

        keep = []
        while len(order) > 0:
            # Take element with highest score
            i = order[0]
            keep.append(i)

            if len(order) == 1:
                break

            # Calculate IoU with all other boxes OF THE SAME CLASS
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])

            w = np.maximum(0.0, xx2 - xx1)
            h = np.maximum(0.0, yy2 - yy1)
            intersection = w * h

            # IoU = intersection / union
            iou = intersection / (areas[i] + areas[order[1:]] - intersection)

            # Keep only boxes with IoU below threshold
            inds = np.where(iou <= iou_threshold)[0]
            order = order[inds + 1]

        # Map back to original indices
        class_keep_indices = class_indices[keep]
        all_keep_indices.extend(class_keep_indices.tolist())

    # Sort by original order to maintain consistency
    all_keep_indices = sorted(all_keep_indices)

    # Return only the kept boxes
    bboxs_filtered = bboxs[all_keep_indices].tolist()
    labels_filtered = labels[all_keep_indices].tolist()
    scores_filtered = scores[all_keep_indices].tolist()

    return bboxs_filtered, labels_filtered, scores_filtered

def rectangles_overlap(rect1, rect2):
    """Check if two rectangles overlap."""
    x1_min, y1_min, x1_max, y1_max = rect1
    x2_min, y2_min, x2_max, y2_max = rect2
    
    return not (x1_max < x2_min or x2_max < x1_min or 
                y1_max < y2_min or y2_max < y1_min)

def draw_detections(img, detections):
    occupied_regions = [] 
    
    for detection in detections:
        x1, y1, x2, y2 = map(int, detection.bbox)
        
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        text = f"{detection.label}: {detection.score:.2f}"
        (text_width, text_height), baseline = cv2.getTextSize(
            text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2
        )
        
        positions = [
            (x1, y1 - text_height - 5),         
            (x1, y2 + text_height + 5),       
            (x2 + 5, y1),                    
            (x1 - text_width - 5, y1),        
            (x1, y1 + text_height + 5),          
            ((x1 + x2) // 2 - text_width // 2, y1 - text_height - 5)  
        ]
        
        final_pos = positions[0]  
        for pos_x, pos_y in positions:
            text_rect = (
                pos_x, 
                pos_y - text_height - baseline - 5,
                pos_x + text_width,
                pos_y
            )
            
            overlaps = False
            for occupied in occupied_regions:
                if rectangles_overlap(text_rect, occupied):
                    overlaps = True
                    break
            
            if (text_rect[0] >= 0 and text_rect[1] >= 0 and 
                text_rect[2] < img.shape[1] and text_rect[3] < img.shape[0] and 
                not overlaps):
                final_pos = (pos_x, pos_y)
                occupied_regions.append(text_rect)
                break

        text_x, text_y = final_pos
        cv2.rectangle(
            img, 
            (text_x, text_y - text_height - baseline - 5), 
            (text_x + text_width, text_y), 
            (0, 255, 0), 
            -1
        )
        cv2.putText(
            img, 
            text, 
            (text_x, text_y - 5), 
            cv2.FONT_HERSHEY_SIMPLEX, 
            0.5, 
            (0, 0, 0), 
            2
        )
    
    path_file = os.path.dirname(os.path.abspath(__file__))
    cv2.imwrite(os.path.join(path_file, "../assets/debug_detections.jpg"), img)
    return img

def compute_iou_3d(bbox1, bbox2, min_size=0.01):
    """
    Compute IoU 3D between two bounding boxes with minimum size expansion.
    Args:
        bbox1, bbox2: dict with keys 'x_min', 'y_min', '
            'z_min', 'x_max', 'y_max', 'z_max'
        min_size: minimum size for each dimension
    Returns:
        iou: float between 0 and 1  
    """

    # Espandi bbox se troppo sottili
    def expand_if_needed(bbox, min_size):
        expanded = bbox.copy()
        for axis in ['x', 'y', 'z']:
            size = bbox[f"{axis}_max"] - bbox[f"{axis}_min"]
            if size < min_size:
                center = (bbox[f"{axis}_min"] + bbox[f"{axis}_max"]) / 2
                expanded[f"{axis}_min"] = center - min_size / 2
                expanded[f"{axis}_max"] = center + min_size / 2
        return expanded

    bbox1_exp = expand_if_needed(bbox1, min_size)
    bbox2_exp = expand_if_needed(bbox2, min_size)

    # Calcola intersezione
    x_inter_min = max(bbox1_exp["x_min"], bbox2_exp["x_min"])
    x_inter_max = min(bbox1_exp["x_max"], bbox2_exp["x_max"])
    y_inter_min = max(bbox1_exp["y_min"], bbox2_exp["y_min"])
    y_inter_max = min(bbox1_exp["y_max"], bbox2_exp["y_max"])
    z_inter_min = max(bbox1_exp["z_min"], bbox2_exp["z_min"])
    z_inter_max = min(bbox1_exp["z_max"], bbox2_exp["z_max"])

    if x_inter_max < x_inter_min or y_inter_max < y_inter_min or z_inter_max < z_inter_min:
        return 0.0

    inter_volume = (x_inter_max - x_inter_min) * (y_inter_max - y_inter_min) * (z_inter_max - z_inter_min)

    volume1 = (bbox1_exp["x_max"] - bbox1_exp["x_min"]) * (bbox1_exp["y_max"] - bbox1_exp["y_min"]) * (bbox1_exp["z_max"] - bbox1_exp["z_min"])
    volume2 = (bbox2_exp["x_max"] - bbox2_exp["x_min"]) * (bbox2_exp["y_max"] - bbox2_exp["y_min"]) * (bbox2_exp["z_max"] - bbox2_exp["z_min"])

    union_volume = volume1 + volume2 - inter_volume

    if union_volume == 0:
        return 0.0

    return inter_volume / union_volume


def bbox_to_dict(bbox):
    if bbox is None:
        return None
    return {
        "x_min": float(bbox.get("x_min")),
        "x_max": float(bbox.get("x_max")),
        "y_min": float(bbox.get("y_min")),
        "y_max": float(bbox.get("y_max")),
        "z_min": float(bbox.get("z_min")),
        "z_max": float(bbox.get("z_max")),
    }

def object_to_dict( obj: Object) -> dict:
    return {
        "label": getattr(obj, "label", ""),
        "color": getattr(obj, "color", ""),
        "material": getattr(obj, "material", ""),
        "shape": getattr(obj, "shape", ""),
        "description": getattr(obj, "description", ""),
        "centroid": getattr(obj, "centroid", None),
        "bbox": bbox_to_dict(getattr(obj, "bbox", None)),
    }