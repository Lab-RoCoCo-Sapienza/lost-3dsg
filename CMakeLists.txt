cmake_minimum_required(VERSION 3.8)
project(lost3dsg)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
  set(CMAKE_CXX_STANDARD 17)
  set(CMAKE_CXX_STANDARD_REQUIRED ON)
endif()

# find dependencies
find_package(ament_cmake_python REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rclpy REQUIRED)
find_package(std_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(visualization_msgs REQUIRED)
find_package(rosidl_default_generators REQUIRED)

# Generate custom messages
rosidl_generate_interfaces(${PROJECT_NAME}
  "msg/Centroid.msg"
  "msg/CentroidArray.msg"
  "msg/Bbox3d.msg"
  "msg/Bbox3dArray.msg"
  "msg/ObjectDescription.msg"
  "msg/ObjectDescriptionArray.msg"
  DEPENDENCIES std_msgs sensor_msgs geometry_msgs
)

# Install Python scripts
install(PROGRAMS
  src/perception_module/perception.py
  src/perception_module/cv_utils.py
  src/perception_module/utils.py
  src/perception_module/object_manager.py
  DESTINATION lib/${PROJECT_NAME}
)

# Install data files (text prompts)
install(FILES
  src/perception_module/object_identification_prompt.txt
  src/perception_module/visual_prompt.txt
  DESTINATION lib/${PROJECT_NAME}
)


# Install launch files
install(DIRECTORY launch/
  DESTINATION share/${PROJECT_NAME}/launch
)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()
