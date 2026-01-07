#!/usr/bin/env python3
import os
import numpy as np
from sensor_msgs.msg import CameraInfo
from visualization_msgs.msg import Marker, MarkerArray
from matplotlib.colors import to_rgb
import cv2
from sensor_msgs.msg import PointCloud2
from sensor_msgs_py import point_cloud2
from std_msgs.msg import Header, String, ColorRGBA
from sensor_msgs.msg import PointField
import json
from geometry_msgs.msg import Point
from utils import statistical_outlier_removal, get_distinct_color
import struct
from openai import OpenAI
import base64
from rclpy.duration import Duration
from rclpy.time import Time
from rclpy.duration import Duration as ROS2Duration
from rclpy.qos import QoSProfile, DurabilityPolicy, ReliabilityPolicy

file_path = os.path.abspath(__file__)

with open(os.path.join(os.path.dirname(file_path), "api.txt"), "r") as f:
    api_key = f.read().strip()

client = OpenAI(api_key=api_key)

def _get_R_and_T(trans):
    Tx_base = trans.transform.translation.x
    Ty_base = trans.transform.translation.y
    Tz_base = trans.transform.translation.z
    T = np.array([Tx_base, Ty_base, Tz_base])
    # Quaternion coordinates
    qx = trans.transform.rotation.x
    qy = trans.transform.rotation.y
    qz = trans.transform.rotation.z
    qw = trans.transform.rotation.w
    
    # Rotation matrix
    R = 2*np.array([[pow(qw,2) + pow(qx,2) - 0.5, qx*qy-qw*qz, qw*qy+qx*qz],[qw*qz+qx*qy, pow(qw,2) + pow(qy,2) - 0.5, qy*qz-qw*qx],[qx*qz-qw*qy, qw*qx+qy*qz, pow(qw,2) + pow(qz,2) - 0.5]])
    return R, T

def _transform_point_xyz(pt_xyz, source_frame, target_frame, timeout=1.0, node=None, tf_buffer=None):
    """Transform a 3D point (numpy array/list len=3) from source_frame to target_frame using TF.

    Args:
        node: ROS2 node instance (required for ROS2)
        tf_buffer: tf2_ros.Buffer instance (optional, will use node's buffer if not provided)
    """
    if target_frame == source_frame:
        return np.array(pt_xyz).reshape(3)

    if node is not None and hasattr(node, 'current_transforms'):
        transforms = node.current_transforms
        if (source_frame, target_frame) in transforms:
            trans = transforms[(source_frame, target_frame)]
            R, T = _get_R_and_T(trans)
            p = np.array(pt_xyz).reshape(3)
            return R.dot(p) + T

    if tf_buffer is None and node is not None:
        tf_buffer = getattr(node, 'tf_buffer', None)

    if tf_buffer is None:
        raise ValueError("tf_buffer or node with tf_buffer attribute required for transform")

    trans = tf_buffer.lookup_transform(target_frame, source_frame, Time(seconds=0), timeout=ROS2Duration(seconds=timeout))

    R, T = _get_R_and_T(trans)
    p = np.array(pt_xyz).reshape(3)
    return R.dot(p) + T

def _clear_markers(topic, node=None, publisher=None):
    """Publish a MarkerArray with DELETEALL to clear previous markers on 'topic'.

    Args:
        node: ROS2 node instance (required if publisher not provided)
        publisher: Existing publisher to use (optional)
    """
    ma = MarkerArray()
    m = Marker()
    m.action = Marker.DELETEALL
    ma.markers.append(m)

    if publisher is not None:
        publisher.publish(ma)
    elif node is not None:
        qos = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        pub = node.create_publisher(MarkerArray, topic, qos)
        import time
        time.sleep(0.05) 
        pub.publish(ma)
    else:
        raise ValueError("Either node or publisher must be provided")

def overlay_mask_on_image(image, mask, color_rgb=(0.0, 1.0, 0.0), alpha=0.5):
    """Apply a semi-transparent mask overlay on a BGR image.

    - image: numpy array BGR uint8 (H,W,3)
    - mask: numpy array 2D uint8 (0 or 255) or bool
    - color_rgb: tuple (r,g,b) in 0..1
    - alpha: blending factor
    """
    mask_bool = (mask > 0)
    color_bgr = np.array([int(c * 255) for c in color_rgb[::-1]], dtype=np.uint8)
    img_pixels = image[mask_bool]
    blended = (img_pixels.astype(float) * (1.0 - alpha) + color_bgr.astype(float) * alpha).astype(np.uint8)
    image[mask_bool] = blended
    return image

def mask_list_to_pointcloud2(masks, depth_image, camera_info, node, labels=None, topic="/pcl_objects", max_points_per_obj=20000, remove_outliers=True, outlier_method='combined', sor_k=30, sor_std=1.0, ror_radius=0.02, ror_min_neighbors=20, depth_z_threshold=2.5, publisher=None, labels_publisher=None):
    """Aggregate masks into a PointCloud2 with an object_id field and publish it (latch=True).

    Args:
        node: ROS2 node instance (required)
        publisher: PointCloud2 publisher (optional, will create if None)
        labels_publisher: String publisher for labels (optional, will create if None)
        remove_outliers: If True, apply filters to remove outliers from the point cloud
        outlier_method: Filtering method - 'statistical', 'radius', 'depth', or 'combined'
        sor_k: Number of neighbors for Statistical Outlier Removal (default: 30)
        sor_std: Standard deviation multiplier for Statistical Outlier Removal (default: 1.0)
        ror_radius: Radius in meters for Radius Outlier Removal (default: 0.02)
        ror_min_neighbors: Minimum number of neighbors for Radius Outlier Removal (default: 20)
        depth_z_threshold: Z-score threshold for depth filtering (default: 2.5)
    """
    node.get_logger().info(f"=== mask_list_to_pointcloud2 START: {len(masks)} masks ===")
    
    if labels is None:
        labels = [f"obj_{i}" for i in range(len(masks))]

    if not isinstance(camera_info, CameraInfo):
        raise TypeError('camera_info must be sensor_msgs.msg.CameraInfo')

    K = camera_info.k
    fx = K[0]
    fy = K[4]
    cx = K[2]
    cy = K[5]
    
    node.get_logger().info(f"Camera intrinsics: fx={fx}, fy={fy}, cx={cx}, cy={cy}")

    current_points = []
    id_to_label = {}

    for obj_idx, mask in enumerate(masks):
        if not isinstance(mask, np.ndarray):
            mask = np.array(mask)
        if mask.ndim == 3:
            mask2d = mask[:, :, 0]
        else:
            mask2d = mask
        mask_bool = mask2d.astype(bool)
        ys, xs = np.nonzero(mask_bool)
        if len(xs) == 0:
            node.get_logger().warn(f"  Empty mask : {obj_idx}, skip")
            continue
        if len(xs) > max_points_per_obj:
            idx = np.linspace(0, len(xs) - 1, max_points_per_obj).astype(int)
            xs = xs[idx]
            ys = ys[idx]
            node.get_logger().info(f"  Mask {obj_idx}: reduced to {len(xs)} points (max={max_points_per_obj})")

        # Use the unique object ID (global counter) to generate consistent colors
        # This will be assigned later, so we'll generate the color after getting unique_obj_id
        # For now, we'll store a placeholder and update it later

        
        depth_values = []
        for x, y in zip(xs, ys):
            z = float(depth_image[y, x])
            if z > 0 and not np.isnan(z):
                depth_values.append(z)

        node.get_logger().info(f"  Mask {obj_idx}: {len(depth_values)} valid depth points out of {len(xs)}")
        
        if len(depth_values) == 0:
            node.get_logger().warn(f"  Mask {obj_idx}: NO valid depth, skip")
            continue

        # median e MAD (Median Absolute Deviation) for robustness
        median_depth = np.median(depth_values)
        mad = np.median(np.abs(np.array(depth_values) - median_depth))

        if mad > 0.001:
            depth_min = median_depth - 4.5 * mad  
            depth_max = median_depth + 4.5 * mad  
        else:
            depth_min = median_depth * 0.5 
            depth_max = median_depth * 1.5  

        node.get_logger().debug(f"Obj {obj_idx}: depth range [{depth_min:.3f}, {depth_max:.3f}] (median={median_depth:.3f}, MAD={mad:.3f})")  

        # ============================================
        # Create 3D points with depth pre-filter
        # ============================================
        points = []
        filtered_count = 0
        for x, y in zip(xs, ys):
            z = float(depth_image[y, x])
            if z <= 0 or np.isnan(z):
                continue

            # Filter anomalous depth
            if z < depth_min or z > depth_max:
                filtered_count += 1
                continue

            X = (x - cx) * z / fx
            Y = (y - cy) * z / fy
            # Store temporary placeholder for RGB (will be updated with unique_obj_id color)
            points.append((float(X), float(Y), float(z), 0.0, int(obj_idx)))

        if filtered_count > 0:
            node.get_logger().info(f"Obj {obj_idx}: pre-filtered {filtered_count} points with anomalous depth") 

        if len(points) == 0:
            continue

        # ============================================
        # Apply outlier filtering if requested
        # ============================================
        # IMPORTANT: Disable filters for small/medium objects and sparse surfaces (< 1500 points)
        # Objects with few points or sparse depth (e.g., tables, bottles) are discarded by filters

        if remove_outliers and len(points) > 20:
            original_count = len(points)

            # Convert list of tuples to numpy array
            points_array = np.array(points)
            xyz = points_array[:, :3]  # Extract only XYZ coordinates

            # Initialize mask with all True
            final_mask = np.ones(len(points), dtype=bool)

            stat_mask = statistical_outlier_removal(xyz, k=sor_k, std_ratio=sor_std)
            final_mask &= stat_mask
            node.get_logger().debug(f"Obj {obj_idx}: statistical filter removed {np.sum(~stat_mask)} points") 
            # Apply the final mask
            points = [points[i] for i in range(len(points)) if final_mask[i]]

            removed_count = original_count - len(points)
            if removed_count > 0:
                node.get_logger().info(f"Object {obj_idx} ({labels[obj_idx]}): removed {removed_count}/{original_count} outlier points ({100*removed_count/original_count:.1f}%)") 

        if len(points) == 0:
            node.get_logger().warn(f"Object {obj_idx} ({labels[obj_idx]}): The filter removed all the points for this object!") 
            continue

        # ============================================
        # Add clean points to the list with INCREASING UNIQUE ID
        # ============================================
        # Use unique ID from global counter instead of obj_idx
        unique_obj_id = node.pcl_object_id_counter

        # Generate color using the same system as centroids and bboxes
        color_rgba = get_distinct_color(unique_obj_id)
        r = int(color_rgba.r * 255) & 0xFF
        g = int(color_rgba.g * 255) & 0xFF
        b = int(color_rgba.b * 255) & 0xFF
        rgb_uint = (r << 16) | (g << 8) | b
        rgb_packed = struct.unpack('f', struct.pack('I', rgb_uint))[0]

        # Transform points from camera frame to map frame
        camera_frame = camera_info.header.frame_id
        points_with_unique_id = []

        for x, y, z, _, _ in points:
            # Transform point from camera frame to map frame
            try:
                point_in_map = _transform_point_xyz((x, y, z), camera_frame, "map", node=node)
                points_with_unique_id.append((
                    float(point_in_map[0]),
                    float(point_in_map[1]),
                    float(point_in_map[2]),
                    rgb_packed,  # Use the color based on unique_obj_id
                    unique_obj_id
                ))
            except Exception as e:
                node.get_logger().warn(f"Failed to transform point ({x}, {y}, {z}) to map frame: {e}")
                continue

        node.get_logger().info(f"  Mask {obj_idx}: {len(points_with_unique_id)} FINAL points transformed to map frame with unique ID {unique_obj_id}")
        current_points.extend(points_with_unique_id)
        id_to_label[unique_obj_id] = labels[obj_idx] if obj_idx < len(labels) else f"obj_{obj_idx}"

        node.pcl_object_id_counter += 1

    node.get_logger().info(f"=== Published {len(current_points)} points from {len(id_to_label)} objects ===")

    if len(current_points) == 0:
        node.get_logger().warn("mask_list_to_pointcloud2: NO POINTS to publish!")
        return

    # ============================================
    # PUBLISH ONLY CURRENT FRAME (no accumulation)
    # This prevents color overlapping when revisiting the same scene
    # ============================================
    points_to_publish = current_points
    labels_to_publish = id_to_label

    node.get_logger().info(f"=== Publishing {len(points_to_publish)} points from {len(labels_to_publish)} objects (current frame only) ===")

    # ============================================
    # Create and publish CURRENT PointCloud2
    # ============================================
    fields = [
        PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
        PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
        PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
        PointField(name='rgb', offset=12, datatype=PointField.FLOAT32, count=1),
        PointField(name='object_id', offset=16, datatype=PointField.INT32, count=1)
    ]

    header = Header()
    header.stamp = node.get_clock().now().to_msg()
    header.frame_id = "map"  # Points are now in map frame after transformation

    cloud_msg = point_cloud2.create_cloud(header, fields, points_to_publish)

    if publisher is None:
        qos = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        publisher = node.create_publisher(PointCloud2, topic, qos)

    publisher.publish(cloud_msg)

    try:
        if labels_publisher is None:
            qos = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
            labels_publisher = node.create_publisher(String, topic + "_labels", qos)

        labels_msg = String()
        labels_msg.data = json.dumps(labels_to_publish)
        labels_publisher.publish(labels_msg)
    except Exception:
        node.get_logger().warn("Unable to publish pcl labels mapping")

    node.get_logger().info(f"mask_list_to_pointcloud2: published {len(labels_to_publish)} objects with {len(points_to_publish)} points (current frame only)")  # ROS2_MIGRATION

def publish_individual_pointclouds_by_id(masks, depth_image, camera_info, node, labels=None, frame_id="camera_link", topic_prefix="/pcl_id", max_points_per_obj=20000, remove_outliers=True, outlier_method='combined', sor_k=15, sor_std=1.5, ror_radius=0.04, ror_min_neighbors=10, depth_z_threshold=2.5, publishers_dict=None, id_counter_start=0, timestamp=None):
    """Publish one PointCloud2 per mask on topics `<topic_prefix>_<id>` (latch=True).

    Args:
        masks: List of masks to process
        depth_image: Depth image array
        camera_info: Camera calibration info
        node: ROS2 node instance (required)
        labels: List of labels for each mask (optional)
        frame_id: Frame ID for the point cloud (default: "camera_link")
        topic_prefix: Prefix for topic names (default: "/pcl_id")
        max_points_per_obj: Maximum points per object (default: 20000)
        remove_outliers: If True, apply filters to remove outliers from the point cloud (default: True)
        outlier_method: Filtering method - 'statistical', 'radius', 'depth', or 'combined' (default: 'combined')
        sor_k: Number of neighbors for Statistical Outlier Removal (default: 15)
        sor_std: Standard deviation multiplier for Statistical Outlier Removal (default: 1.5)
        ror_radius: Radius in meters for Radius Outlier Removal (default: 0.04)
        ror_min_neighbors: Minimum number of neighbors for Radius Outlier Removal (default: 10)
        depth_z_threshold: Z-score threshold for depth filtering (default: 2.5)
        publishers_dict: Dictionary to store/reuse publishers {topic_name: publisher} (optional)
        id_counter_start: Starting counter for unique object IDs across detections (default: 0)
        timestamp: Timestamp to use for messages (default: now())

    Returns:
        int: Number of PointCloud2 messages published.
    """
    if labels is None:
        labels = [f"obj_{i}" for i in range(len(masks))]

    if not isinstance(camera_info, CameraInfo):
        raise TypeError('camera_info must be sensor_msgs.msg.CameraInfo')

    K = camera_info.k
    fx = K[0]
    fy = K[4]
    cx = K[2]
    cy = K[5]

    palette_names = ['red', 'green', 'blue', 'magenta', 'cyan', 'yellow', 'orange', 'purple', 'brown', 'pink']
    palette = [to_rgb(c) for c in palette_names]

    published_count = 0
    for obj_idx, mask in enumerate(masks):
        # Unique ID
        obj_id = id_counter_start + obj_idx
        if not isinstance(mask, np.ndarray):
            mask = np.array(mask)
        if mask.ndim == 3:
            mask2d = mask[:, :, 0]
        else:
            mask2d = mask
        mask_bool = mask2d.astype(bool)
        ys, xs = np.nonzero(mask_bool)
        if len(xs) == 0:
            continue
        if len(xs) > max_points_per_obj:
            idx = np.linspace(0, len(xs) - 1, max_points_per_obj).astype(int)
            xs = xs[idx]
            ys = ys[idx]

        col = palette[obj_idx % len(palette)]
        r = int(col[0] * 255) & 0xFF
        g = int(col[1] * 255) & 0xFF
        b = int(col[2] * 255) & 0xFF
        rgb_uint = (r << 16) | (g << 8) | b
        rgb_packed = struct.unpack('f', struct.pack('I', rgb_uint))[0]

        depth_values = []
        for x, y in zip(xs, ys):
            z = float(depth_image[y, x])
            if z > 0 and not np.isnan(z):
                depth_values.append(z)

        if len(depth_values) == 0:
            continue

        median_depth = np.median(depth_values)
        mad = np.median(np.abs(np.array(depth_values) - median_depth))

        # Determine valid depth range for this object
        if mad > 0.001: 
            depth_min = median_depth - 3.5 * mad
            depth_max = median_depth + 3.5 * mad
        else:
            depth_min = median_depth * 0.7
            depth_max = median_depth * 1.3

        node.get_logger().debug(f"Obj {obj_id}: depth range [{depth_min:.3f}, {depth_max:.3f}] (median={median_depth:.3f}, MAD={mad:.3f})")  # ROS2_MIGRATION

        points = []
        filtered_count = 0
        camera_frame = camera_info.header.frame_id

        for x, y in zip(xs, ys):
            z = float(depth_image[y, x])
            if z <= 0 or np.isnan(z):
                continue

            if z < depth_min or z > depth_max:
                filtered_count += 1
                continue

            X = (x - cx) * z / fx
            Y = (y - cy) * z / fy

            # If frame_id is different from the camera frame, transform the point
            if frame_id != camera_frame:
                try:
                    X_tf, Y_tf, Z_tf = _transform_point_xyz(
                        (float(X), float(Y), float(z)),
                        source_frame=camera_frame,
                        target_frame=frame_id,
                        node=node
                    )
                    points.append((float(X_tf), float(Y_tf), float(Z_tf), float(rgb_packed), int(obj_id)))
                except Exception as e:
                    node.get_logger().warn(f"Trasformazione punto fallita: {e}")
                    continue
            else:
                points.append((float(X), float(Y), float(z), float(rgb_packed), int(obj_id)))

        if filtered_count > 0:
            node.get_logger().info(f"Obj {obj_id}: pre-filtered {filtered_count} points with anomalous depth")  # ROS2_MIGRATION

        if len(points) == 0:
            continue

        # Applica filtraggio outlier se richiesto
        if remove_outliers and len(points) > 20:
            original_count = len(points)

            # Converti lista di tuple in array numpy
            points_array = np.array(points)
            xyz = points_array[:, :3]  # Estrai solo coordinate XYZ

            # Inizializza maschera con tutti True
            final_mask = np.ones(len(points), dtype=bool)

            stat_mask = statistical_outlier_removal(xyz, k=sor_k, std_ratio=sor_std)
            final_mask &= stat_mask
            node.get_logger().debug(f"Obj {obj_id}: statistical filter removed {np.sum(~stat_mask)} points")  # ROS2_MIGRATION

            points = [points[i] for i in range(len(points)) if final_mask[i]]

            removed_count = original_count - len(points)
            if removed_count > 0:
                node.get_logger().info(f"Object {obj_idx} ({labels[obj_idx]}): removed {removed_count}/{original_count} outlier points ({100*removed_count/original_count:.1f}%)")  # ROS2_MIGRATION

        if len(points) == 0:
            node.get_logger().warn(f"Object {obj_idx} ({labels[obj_idx]}): no points left after outlier removal!")  # ROS2_MIGRATION
            continue

        fields = [
            PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
            PointField(name='rgb', offset=12, datatype=PointField.FLOAT32, count=1),
            PointField(name='object_id', offset=16, datatype=PointField.INT32, count=1)
        ]

        header = Header()
        header.stamp = timestamp if timestamp is not None else node.get_clock().now().to_msg()
        header.frame_id = frame_id

        cloud_msg = point_cloud2.create_cloud(header, fields, points)
        label_sanitized = labels[obj_idx].replace(' ', '_').replace('-', '_')
        label_sanitized = ''.join(c if c.isalnum() or c == '_' else '_' for c in label_sanitized)
        topic_name = f"{label_sanitized}_{obj_id}"

        if publishers_dict is not None and topic_name in publishers_dict:
            pub = publishers_dict[topic_name]
        else:
            qos = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
            pub = node.create_publisher(PointCloud2, topic_name, qos)
            if publishers_dict is not None:
                publishers_dict[topic_name] = pub

        pub.publish(cloud_msg)
        published_count += 1

    node.get_logger().info(f"publish_individual_pointclouds_by_id: published {published_count} PointCloud2")
    return published_count
	
def points_list_to_rviz_3d(points, node, centroid_marker_pub=None, labels=None, frame_id="map", topic="/centroid_markers", marker_scale=0.06):
    """
    Visualize a list of 3D points in RViz as spheres.

    Args:
        node: ROS2 node instance (required)
        centroid_marker_pub: Existing publisher (optional, creates a new one if None)
        points: list of tuples (x, y, z)
        labels: list of corresponding labels (optional)
        frame_id: reference frame for the markers
        topic: ROS topic to publish the markers on
        marker_scale: size of the spheres in RViz
    """
    if centroid_marker_pub is None:
        qos = QoSProfile(depth=10, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        centroid_marker_pub = node.create_publisher(MarkerArray, topic, qos)

    marker_array = MarkerArray()

    if not hasattr(node, '_centroid_marker_id_counter'):
        node._centroid_marker_id_counter = 0

    color_text = ColorRGBA(r=1.0, g=1.0, b=1.0, a=1.0)

    for i, point in enumerate(points):
        if point is None:
            continue

        x, y, z = point
        label = labels[i] if labels and i < len(labels) else f"obj_{i}"

        # Generate distinct color for each marker
        color = get_distinct_color(node._centroid_marker_id_counter)

        # Use unique IDs for sphere and text markers
        sphere_id = node._centroid_marker_id_counter
        text_id = node._centroid_marker_id_counter + 100000  # Offset grande per evitare sovrapposizioni

        node._centroid_marker_id_counter += 1

        # Sphere marker for the 3D point
        marker = Marker()
        marker.header.frame_id = frame_id
        marker.header.stamp = node.get_clock().now().to_msg()
        marker.ns = "centroid_spheres"
        marker.id = sphere_id
        marker.type = Marker.SPHERE
        marker.action = Marker.ADD
        marker.pose.position.x = x
        marker.pose.position.y = y
        marker.pose.position.z = z
        marker.pose.orientation.x = 0.0
        marker.pose.orientation.y = 0.0
        marker.pose.orientation.z = 0.0
        marker.pose.orientation.w = 1.0
        marker.scale.x = marker.scale.y = marker.scale.z = marker_scale
        marker.color = color
        marker.lifetime = Duration(seconds=0).to_msg()  # Permanente

        # Text marker for the label
        text_marker = Marker()
        text_marker.header.frame_id = frame_id
        text_marker.header.stamp = node.get_clock().now().to_msg()
        text_marker.ns = "centroid_labels"
        text_marker.id = text_id
        text_marker.type = Marker.TEXT_VIEW_FACING
        text_marker.action = Marker.ADD
        text_marker.pose.position.x = x
        text_marker.pose.position.y = y
        text_marker.pose.position.z = z + marker_scale * 1.5
        text_marker.pose.orientation.x = 0.0
        text_marker.pose.orientation.y = 0.0
        text_marker.pose.orientation.z = 0.0
        text_marker.pose.orientation.w = 1.0
        text_marker.scale.z = marker_scale * 1.2
        text_marker.color = color_text
        # Replace spaces with newlines for better readability in RViz
        text_marker.text = label.replace(' ', '\n')
        text_marker.lifetime = Duration(seconds=0).to_msg()

        marker_array.markers.append(marker)
        marker_array.markers.append(text_marker)

    if len(marker_array.markers) > 0:
        centroid_marker_pub.publish(marker_array)

def init_bbox_publisher(node):
    """
    Initialize publishers for bbox and centroid markers.

    Args:
        node: ROS2 node instance (required)

    Returns:
        tuple: (bbox_marker_pub, centroid_marker_pub)
    """
    # ROS2_MIGRATION: Usa QoS compatibile con RViz (BEST_EFFORT + VOLATILE)
    qos = QoSProfile(
        depth=10, 
        durability=DurabilityPolicy.VOLATILE,
        reliability=ReliabilityPolicy.BEST_EFFORT
    )

    bbox_marker_pub = node.create_publisher(MarkerArray, '/bbox_marker', qos)
    centroid_marker_pub = node.create_publisher(MarkerArray, '/centroid_markers', qos)

    import time
    time.sleep(0.5)  # ROS2_MIGRATION: Dai tempo al publisher di registrarsi

    return bbox_marker_pub, centroid_marker_pub
  
def mask_list_to_centroid_and_bbox(mask_list, labels, depth_image, camera_info, node, bbox_marker_pub=None, centroid_marker_pub=None, max_points_per_obj=20000,
                                   remove_outliers=True, sor_k=30, sor_std=1.0):
    """
    Calculate centroids and 3D bounding boxes ONLY on filtered points (like mask_list_to_pointcloud2).
    The results can be visualized in RViz using the provided publishers.
    The filter operation ensures that centroids and bounding boxes are computed only on reliable 3D points.

    Args:
        mask_list: List of binary masks for detected objects
        labels: List of labels corresponding to each mask
        depth_image: Depth image array
        camera_info: Camera calibration info
        node: ROS2 node instance (required)
        bbox_marker_pub: Publisher for bounding box markers (optional)
        centroid_marker_pub: Publisher for centroid markers (optional)
        max_points_per_obj: Maximum points to consider per object (default: 20000)
        remove_outliers: If True, apply outlier removal filters (default: True)
        sor_k: Number of neighbors for Statistical Outlier Removal (default: 30)
        sor_std: Standard deviation multiplier for Statistical Outlier Removal (default: 1.0
    """

    K = camera_info.k
    fx, fy, cx, cy = K[0], K[4], K[2], K[5]
    
    centroids_3d = []
    bboxes_3d = []
    all_markers = []

    # Initialize unique ID counter for bbox markers
    if not hasattr(node, '_bbox_marker_id_counter'):
        node._bbox_marker_id_counter = 0

    for mask_idx, mask in enumerate(mask_list):
        if not isinstance(mask, np.ndarray):
            mask = np.array(mask)
        if mask.ndim == 3:
            mask = mask[:, :, 0]

        mask_bool = mask.astype(bool)
        ys, xs = np.nonzero(mask_bool)

        if len(xs) == 0:
            node.file_logger.warning(f"Mask {mask_idx}: discarded - empty mask (no pixels)")
            centroids_3d.append(None)
            bboxes_3d.append(None)
            continue

        # Log start of mask processing
        node.file_logger.info(f"Mask {mask_idx}: start processing, total pixels={len(xs)}")

        if len(xs) > max_points_per_obj:
            idxs = np.linspace(0, len(xs) - 1, max_points_per_obj).astype(int)
            xs, ys = xs[idxs], ys[idxs]

        depth_values = []
        for x, y in zip(xs, ys):
            z = float(depth_image[y, x])
            if z > 0 and not np.isnan(z):
                depth_values.append(z)

        if len(depth_values) == 0:
            node.file_logger.warning(f"Mask {mask_idx}: discarded - no valid depth pixels (all ≤0 or NaN). Mask pixels: {len(xs)}")
            centroids_3d.append(None)
            bboxes_3d.append(None)
            continue

        # Use median and MAD (identical to mask_list_to_pointcloud2)
        median_depth = np.median(depth_values)
        mad = np.median(np.abs(np.array(depth_values) - median_depth))

        if mad > 0.001:
            depth_min = median_depth - 4.5 * mad
            depth_max = median_depth + 4.5 * mad
        else:
            depth_min = median_depth * 0.5
            depth_max = median_depth * 1.5

        # Create an auxiliary list of 3D points with depth pre-filter
        points_3d = []
        for x, y in zip(xs, ys):
            z = float(depth_image[y, x])
            if z <= 0 or np.isnan(z):
                continue
            if z < depth_min or z > depth_max:
                continue  # Anomalous depth, skip

            X = (x - cx) * z / fx
            Y = (y - cy) * z / fy
            points_3d.append([X, Y, z])

        if len(points_3d) == 0:
            node.file_logger.warning(f"Mask {mask_idx}: discarded - all points filtered out by MAD pre-filter. Initial valid depth points: {len(depth_values)}, median={median_depth:.3f}m, MAD={mad:.3f}m, range=[{depth_min:.3f}, {depth_max:.3f}]")
            centroids_3d.append(None)
            bboxes_3d.append(None)
            continue

        points_3d = np.array(points_3d)

        # ============================================
        # APPLY OUTLIER FILTERING IF REQUESTED
        # ============================================
        # IMPORTANT: Disable filters for small/medium objects and sparse surfaces (< 1500 points)
        # During the test objects with few points or sparse depth (e.g., tables, bottles) are skipped by the filters
        # This ensures centroids and bboxes are computed on reliable points only (check this numbers in case of issues)

        if remove_outliers and len(points_3d) > 20:
            
            # Large object - apply filters normally
            final_mask = np.ones(len(points_3d), dtype=bool)

            
            stat_mask = statistical_outlier_removal(points_3d, k=sor_k, std_ratio=sor_std)
            final_mask &= stat_mask


            points_3d = points_3d[final_mask]
            node.file_logger.info(f"Mask {mask_idx}: after outlier filters {len(points_3d)} points remain")

        if len(points_3d) == 0:
            node.get_logger().warn(f"Mask {mask_idx}: no points after outlier filtering!")
            centroids_3d.append(None)
            bboxes_3d.append(None)
            continue

        # Depth statistics of the points
        depths = points_3d[:, 2]  # Z is the depth
        node.file_logger.info(f"  Depth stats: min={depths.min():.3f}m, max={depths.max():.3f}m, mean={depths.mean():.3f}m, std={depths.std():.3f}m")

        # ============================================
        # CALCULATE CENTROID AND BBOX ON CLEAN POINTS
        # ============================================
        camera_frame = camera_info.header.frame_id
        centroid = np.mean(points_3d, axis=0)
        centroids_3d.append(tuple(centroid))

        # Transform centroids in map frame and publish marker
        if centroid_marker_pub is not None:
            centroid_in_map = _transform_point_xyz(tuple(centroid), camera_frame, "map", node=node)
            points = [(centroid_in_map[0], centroid_in_map[1], centroid_in_map[2])]
            # Pass only the current object's label, not the entire labels array
            current_label = [labels[mask_idx]] if labels and mask_idx < len(labels) else [f"obj_{mask_idx}"]
            points_list_to_rviz_3d(points, node, centroid_marker_pub=centroid_marker_pub, labels=current_label, frame_id="map", marker_scale=0.05)

        x_min, x_max = points_3d[:, 0].min(), points_3d[:, 0].max()
        y_min, y_max = points_3d[:, 1].min(), points_3d[:, 1].max()
        z_min, z_max = points_3d[:, 2].min(), points_3d[:, 2].max()


        # Log dimension of bbox in camera frame
        bbox_size = (x_max - x_min, y_max - y_min, z_max - z_min)
        node.file_logger.info(f"  BBox camera frame size: X={bbox_size[0]:.3f}, Y={bbox_size[1]:.3f}, Z={bbox_size[2]:.3f}")

        # ============================================
        # TRANSFORM TO MAP FRAME AND CREATE MARKER
        # ============================================
        bbox_points = [
            (x_min, y_min, z_min), (x_max, y_min, z_min),
            (x_max, y_max, z_min), (x_min, y_max, z_min),
            (x_min, y_min, z_max), (x_max, y_min, z_max),
            (x_max, y_max, z_max), (x_min, y_max, z_max),
        ]

        # Transform bbox vertices to map frame for static room coordinates
        try:
            bbox_points_map = [
                _transform_point_xyz(p, camera_frame, "map", node=node)
                for p in bbox_points
            ]
            
            # Calculate bbox in MAP coordinates from transformed points
            bbox_points_map_array = np.array(bbox_points_map)
            bbox_dict = {
                "x_min": float(bbox_points_map_array[:, 0].min()),
                "x_max": float(bbox_points_map_array[:, 0].max()),
                "y_min": float(bbox_points_map_array[:, 1].min()),
                "y_max": float(bbox_points_map_array[:, 1].max()),
                "z_min": float(bbox_points_map_array[:, 2].min()),
                "z_max": float(bbox_points_map_array[:, 2].max())
            }

            # Log dimension of bbox in MAP frame
            map_bbox_size = (bbox_dict["x_max"] - bbox_dict["x_min"],
                           bbox_dict["y_max"] - bbox_dict["y_min"],
                           bbox_dict["z_max"] - bbox_dict["z_min"])
            node.file_logger.info(f"  BBox MAP frame size: X={map_bbox_size[0]:.3f}, Y={map_bbox_size[1]:.3f}, Z={map_bbox_size[2]:.3f}")

            bboxes_3d.append(bbox_dict)
            
        except Exception as e:
            node.get_logger().warn(f"Impossible to transform bbox to map frame for mask {mask_idx}: {e}. Using camera coordinates.")
            # Fallback: use camera coordinates
            bbox_dict = {
                "x_min": float(x_min), "x_max": float(x_max),
                "y_min": float(y_min), "y_max": float(y_max),
                "z_min": float(z_min), "z_max": float(z_max)
            }
            bboxes_3d.append(bbox_dict)
            continue

        # Use color based on global counter for consistency
        bbox_color = get_distinct_color(node._bbox_marker_id_counter)

        marker = Marker()
        marker.header.frame_id = "map"
        marker.header.stamp = node.get_clock().now().to_msg()
        marker.ns = "bbox_markers"
        marker.id = node._bbox_marker_id_counter  # Use unique ID
        marker.type = Marker.SPHERE_LIST
        marker.action = Marker.ADD
        marker.scale.x = marker.scale.y = marker.scale.z = 0.02  # Smaller spheres for bbox corners
        marker.color = bbox_color
        marker.lifetime = Duration(seconds=0).to_msg()  # Permanent

        for p in bbox_points_map:
            pt = Point()
            pt.x, pt.y, pt.z = p
            marker.points.append(pt)

        all_markers.append(marker)

        node._bbox_marker_id_counter += 1

    # Publish all bbox markers at once
    if bbox_marker_pub is not None and len(all_markers) > 0:
        array = MarkerArray()
        array.markers = all_markers
        bbox_marker_pub.publish(array)
        node.get_logger().info(f"Published {len(all_markers)} bbox markers on /bbox_marker")
    elif bbox_marker_pub is None:
        node.get_logger().warn("Publisher bbox_marker not initialized!")

    return centroids_3d, bboxes_3d

def publish_pov_volume(node, pov_volume, considered_volume_pub=None):
        """
        Publishes the POV (Point of View) volume to RViz.
        This volume represents "what the camera is looking at" in this frame.
        Objects inside this volume that are not seen will be removed.

        Args:
            pov_volume: dict with x_min, x_max, y_min, y_max, z_min, z_max
        """
        marker = Marker()
        marker.header.frame_id = "map"
        marker.header.stamp = node.get_clock().now().to_msg()
        marker.ns = "pov_volume"
        marker.id = 0
        marker.type = Marker.CUBE
        marker.action = Marker.ADD

        marker.pose.position.x = (pov_volume["x_min"] + pov_volume["x_max"]) / 2.0
        marker.pose.position.y = (pov_volume["y_min"] + pov_volume["y_max"]) / 2.0
        marker.pose.position.z = (pov_volume["z_min"] + pov_volume["z_max"]) / 2.0
        marker.pose.orientation.w = 1.0

   
        marker.scale.x = pov_volume["x_max"] - pov_volume["x_min"]
        marker.scale.y = pov_volume["y_max"] - pov_volume["y_min"]
        marker.scale.z = pov_volume["z_max"] - pov_volume["z_min"]

        marker.color.r = 1.0
        marker.color.g = 0.2
        marker.color.b = 0.2
        marker.color.a = 0.15

        marker.lifetime = Duration(seconds=0).to_msg()  # PERMANENTE

        # Marker per il testo
        text_marker = Marker()
        text_marker.header.frame_id = "map"
        text_marker.header.stamp = node.get_clock().now().to_msg()
        text_marker.ns = "pov_label"
        text_marker.id = 1
        text_marker.type = Marker.TEXT_VIEW_FACING
        text_marker.action = Marker.ADD

        text_marker.pose.position.x = marker.pose.position.x
        text_marker.pose.position.y = marker.pose.position.y
        text_marker.pose.position.z = pov_volume["z_max"] + 0.25
        text_marker.pose.orientation.w = 1.0

        text_marker.text = "POV VOLUME"
        text_marker.scale.z = 0.15

        text_marker.color.r = 1.0
        text_marker.color.g = 0.5
        text_marker.color.b = 0.5
        text_marker.color.a = 1.0

        text_marker.lifetime = Duration(seconds=0).to_msg()  # PERMANENTE

        marker_array = MarkerArray()
        marker_array.markers.append(marker)
        marker_array.markers.append(text_marker)

        considered_volume_pub.publish(marker_array)

def publish_persistent_bboxes(node, wm, persistent_bboxes_pub=None):

        # Log dettagliato degli oggetti da pubblicare
        for obj in wm.persistent_perceptions:
            if obj.bbox is not None:
                x_size = obj.bbox["x_max"] - obj.bbox["x_min"]
                y_size = obj.bbox["y_max"] - obj.bbox["y_min"]
                z_size = obj.bbox["z_max"] - obj.bbox["z_min"]

        marker_array = MarkerArray()

        for i, obj in enumerate(wm.persistent_perceptions):
            
            # Marker per il cubo (bounding box)
            marker = Marker()
            marker.header.frame_id = "map"
            # ROS2_MIGRATION
            marker.header.stamp = node.get_clock().now().to_msg()
            marker.ns = "persistent_bboxes"
            marker.id = i * 2  # ID pari per i cubi
            marker.type = Marker.CUBE
            marker.action = Marker.ADD

            # Posizione (centro del bbox)
            marker.pose.position.x = (obj.bbox["x_min"] + obj.bbox["x_max"]) / 2.0
            marker.pose.position.y = (obj.bbox["y_min"] + obj.bbox["y_max"]) / 2.0
            marker.pose.position.z = (obj.bbox["z_min"] + obj.bbox["z_max"]) / 2.0
            marker.pose.orientation.w = 1.0

            # Dimensions
            marker.scale.x = obj.bbox["x_max"] - obj.bbox["x_min"]
            marker.scale.y = obj.bbox["y_max"] - obj.bbox["y_min"]
            marker.scale.z = obj.bbox["z_max"] - obj.bbox["z_min"]

            # Color (semi-transparent green)
            marker.color.r = 0.0
            marker.color.g = 1.0
            marker.color.b = 0.0
            marker.color.a = 0.3

            # ROS2_MIGRATION: Duration(0) significa permanente
            marker.lifetime = Duration(seconds=0).to_msg()
            marker_array.markers.append(marker)

            # Marker for text (label)
            text_marker = Marker()
            text_marker.header.frame_id = "map"
            # ROS2_MIGRATION
            text_marker.header.stamp = node.get_clock().now().to_msg()
            text_marker.ns = "persistent_labels"
            text_marker.id = i * 2 + 1  # Odd ID for texts
            text_marker.type = Marker.TEXT_VIEW_FACING
            text_marker.action = Marker.ADD

            # Position (above the bbox)
            text_marker.pose.position.x = marker.pose.position.x
            text_marker.pose.position.y = marker.pose.position.y
            text_marker.pose.position.z = obj.bbox["z_max"] + 0.1
            text_marker.pose.orientation.w = 1.0

            # Text - Replace spaces with newlines for better readability
            text_marker.text = obj.label.replace(' ', '\n')
            text_marker.scale.z = 0.1  # Text height

            # Color (white)
            text_marker.color.r = 1.0
            text_marker.color.g = 1.0
            text_marker.color.b = 1.0
            text_marker.color.a = 1.0

            # ROS2_MIGRATION
            text_marker.lifetime = Duration(seconds=0).to_msg()
            marker_array.markers.append(text_marker)

        # Publish
        persistent_bboxes_pub.publish(marker_array)

def publish_uncertain_bboxes(node, uncertain_objects, uncertain_bbox_pub):
        """
        Publish bounding boxes of uncertain objects (ORANGE) on RViz.
        These objects represent old positions of objects that have been moved > 80cm.
        """
        if not uncertain_objects:
            return


        marker_array = MarkerArray()

        for i, obj in enumerate(uncertain_objects):
            if obj.bbox is None:
                continue

            # Marker for cube (ORANGE bounding box)
            marker = Marker()
            marker.header.frame_id = "map"
            marker.header.stamp = node.get_clock().now().to_msg()
            marker.ns = "uncertain_bboxes"
            marker.id = i * 2  # Even ID for cubes
            marker.type = Marker.CUBE
            marker.action = Marker.ADD

            # Position (bbox center)
            marker.pose.position.x = (obj.bbox["x_min"] + obj.bbox["x_max"]) / 2.0
            marker.pose.position.y = (obj.bbox["y_min"] + obj.bbox["y_max"]) / 2.0
            marker.pose.position.z = (obj.bbox["z_min"] + obj.bbox["z_max"]) / 2.0
            marker.pose.orientation.w = 1.0

            # Dimensions
            marker.scale.x = obj.bbox["x_max"] - obj.bbox["x_min"]
            marker.scale.y = obj.bbox["y_max"] - obj.bbox["y_min"]
            marker.scale.z = obj.bbox["z_max"] - obj.bbox["z_min"]

            # Color (semi-transparent ORANGE)
            marker.color.r = 1.0
            marker.color.g = 0.6
            marker.color.b = 0.0
            marker.color.a = 0.4

            marker.lifetime = Duration(seconds=0).to_msg()
            marker_array.markers.append(marker)

            # Marker for text (label)
            text_marker = Marker()
            text_marker.header.frame_id = "map"
            text_marker.header.stamp = node.get_clock().now().to_msg()
            text_marker.ns = "uncertain_labels"
            text_marker.id = i * 2 + 1  # Odd ID for texts
            text_marker.type = Marker.TEXT_VIEW_FACING
            text_marker.action = Marker.ADD

            # Position (above the bbox)
            text_marker.pose.position.x = marker.pose.position.x
            text_marker.pose.position.y = marker.pose.position.y
            text_marker.pose.position.z = obj.bbox["z_max"] + 0.1
            text_marker.pose.orientation.w = 1.0

            # Text - Replace spaces with newlines for better readability
            label_with_newlines = obj.label.replace(' ', '\n')
            text_marker.text = f"{label_with_newlines}\n[UNCERTAIN]"
            text_marker.scale.z = 0.1  # Text height

            # Color (ORANGE)
            text_marker.color.r = 1.0
            text_marker.color.g = 0.6
            text_marker.color.b = 0.0
            text_marker.color.a = 1.0

            text_marker.lifetime = Duration(seconds=0).to_msg()
            marker_array.markers.append(text_marker)

        # Publish
        uncertain_bbox_pub.publish(marker_array)
        


def vlm_call(prompt, encoded_image):
    agent = client.chat.completions.create(
        model="gpt-5-mini-2025-08-07",
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{encoded_image}"}
                    }
                ],
            }
        ]
    )
    return agent.choices[0].message.content


def numpy_to_base64(img, fmt='.png'):
    _, buffer = cv2.imencode(fmt, img)
    return base64.b64encode(buffer).decode('utf-8')