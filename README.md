<div align="center">
<img src="/assets/logo.png" width=60%>
<br>

<h3>LOST-3DSG: Lightweight Open-Vocabulary 3D Scene Graphs with Semantic Tracking in Dynamic Environments</h3>
  
<a href="https://www.linkedin.com/in/sara-micol-ferraina-0a51173a1/">Sara Micol Ferraina</a><sup><span>*1</span></sup>,
<a href="https://scholar.google.com/citations?user=sk3SpmUAAAAJ&hl=it&oi=ao/">Michele Brienza</a><sup><span>*1</span></sup>,
<a href="https://www.linkedin.com/in/fra-arg/">Francesco Argenziano</a><sup><span>1</span></sup>,
<a href="https://scholar.google.com/citations?user=XLcFkmUAAAAJ&hl=it">Emanuele Musumeci</a><sup><span>1</span></sup>,
<a href="https://scholar.google.com/citations?user=Y8LuLfoAAAAJ&hl=it&oi=ao">Vincenzo Suriani</a><sup><span>1</span></sup>,
<a href="https://scholar.google.com/citations?user=_90LQXQAAAAJ&hl=it&oi=ao">Domenico D. Bloisi</a><sup><span>2</span></sup>,
<a href="https://scholar.google.com/citations?user=xZwripcAAAAJ&hl=it&oi=ao">Daniele Nardi</a><sup><span>1</span></sup>
</br>

<span style="font-size: 0.8em;"><sup>*</sup>Authors contributed equally</span>

<sup>1</sup> Department of Computer, Control and Management Engineering, Sapienza University of Rome, Rome, Italy,
<sup>2</sup> International University of Rome UNINT, Rome, Italy

<div>

[![arxiv paper](https://img.shields.io/badge/Project-Website-blue)](https://lab-rococo-sapienza.github.io/lost-3dsg/)
[![arxiv paper](https://img.shields.io/badge/arXiv-PDF-red)](https://lab-rococo-sapienza.github.io/lost-3dsg/)
[![license](https://img.shields.io/badge/License-Apache_2.0-yellow)](LICENSE)

</div>

<img src="/assets/architecture.png" width=100%>

</div>

## Installation

### Prerequisites

- Ubuntu 22.04 (recommended)
- ROS 2 Humble
- Python 3.10+
- CUDA-capable GPU (recommended for optimal performance)

### Step 1: Install ROS 2 Humble

If you haven't already installed ROS 2 Humble, follow the [official installation guide](https://docs.ros.org/en/humble/Installation.html).

```bash
# Quick install (Ubuntu 22.04)
sudo apt update
sudo apt install ros-humble-desktop
source /opt/ros/humble/setup.bash
```

### Step 2: Install System Dependencies

Install the required ROS 2 packages:

```bash
sudo apt install -y \
  ros-humble-rclpy \
  ros-humble-cv-bridge \
  ros-humble-sensor-msgs \
  ros-humble-visualization-msgs \
  ros-humble-tf2-ros \
  ros-humble-geometry-msgs \
  python3-pip \
  python3-colcon-common-extensions
```

### Step 3: Clone the Repository in workspace (in the example ~/ros2_ws/src)

```bash
mkdir -p ~/ros2_ws/src
cd ~/ros2_ws/src
git clone https://github.com/Lab-RoCoCo-Sapienza/lost-3dsg/
cd lost-3dsg
```

### Step 4: Install Python Dependencies

Install the required Python packages:

```bash
pip3 install -r requirements.txt
```

**Note:** If you have a CUDA-capable GPU, ensure you have the appropriate CUDA toolkit installed for PyTorch and ONNX Runtime GPU support.

### Step 5: Build the Package

```bash
cd ~/ros2_ws
colcon build --packages-select lost3dsg
source install/setup.bash
```

### Step 6: Verify Installation

Check if the package is properly installed:

```bash
ros2 pkg list | grep lost3dsg
```

## Usage

### Configure Camera Topics

Before running the perception module, you need to configure the ROS 2 topic names to match your robot's camera topics. Edit [src/perception_module/utils.py](src/perception_module/utils.py) in the `SyncedCameraData.__init__()` method (around line 50):

```python
# Change these topic names to match your robot
node.create_subscription(Image, '/head_front_camera/rgb/image_raw', self._rgb_callback, qos_sensor)
node.create_subscription(Image, '/head_front_camera/depth/image_raw', self._depth_callback, qos_sensor)
node.create_subscription(CameraInfo, '/head_front_camera/rgb/camera_info', self._camera_info_callback, qos_sensor)
```

**Example for a different robot:**
```python
# For a robot with different camera topics
node.create_subscription(Image, '/camera/color/image_raw', self._rgb_callback, qos_sensor)
node.create_subscription(Image, '/camera/depth/image_raw', self._depth_callback, qos_sensor)
node.create_subscription(CameraInfo, '/camera/color/camera_info', self._camera_info_callback, qos_sensor)
```

You can list your robot's available topics with:
```bash
ros2 topic list | grep camera
```

### Run LOST-3DSG

To run the full pipeline, start the following two modules in separate terminals. The Scene Updater Module subscribes to the Perception module's outputs to build the scene graph.

### Running the Perception Module

The perception module performs real-time object detection and segmentation using Vision-Language Models (VLMs). It identifies objects in the camera view, generates 3D bounding boxes, extracts visual features (color, material, shape), and publishes point clouds and semantic descriptions.

```bash
# Source the workspace
source ~/ros2_ws/install/setup.bash

# Run the perception node
ros2 run lost3dsg perception.py
```

### Running the Scene Update Module

The object manager maintains a persistent 3D scene graph by tracking detected objects across frames. It handles object association, movement detection, and spatial reasoning to build a consistent world model. This module distinguishes between new objects, moved objects, and objects that have left the field of view.

```bash
# Source the workspace
source ~/ros2_ws/install/setup.bash

# Run the object manager node
ros2 run lost3dsg object_manager.py
```
## Configuration

Configuration files and prompts are located in:
- Object identification prompt: `src/perception_module/object_identification_prompt.txt`
- Visual prompt: `src/perception_module/visual_prompt.txt`

## Troubleshooting

### CUDA/GPU Issues

If you encounter GPU-related errors:
- Verify CUDA installation: `nvidia-smi`
- Install CPU-only versions by modifying `requirements.txt` (remove `onnxruntime-gpu`)

### Import Errors

If you get module import errors:
```bash
# Ensure ROS 2 environment is sourced
source /opt/ros/humble/setup.bash
source ~/ros2_ws/install/setup.bash
```

