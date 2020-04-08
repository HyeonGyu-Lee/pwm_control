cmake_minimum_required(VERSION 2.8.3)
project(pwm_control)

#add_compile_options(-std=c++11)

#set(OpenCV_INCLUDE_DIRS /usr/share/OpenCV/)
#message("OpenCV_DIR=${OpenCV_INCLUDE_DIRS}")
#message("catkin_DIR=${catkin_INCLUDE_DIRS}")
find_package(catkin REQUIRED COMPONENTS
  cv_bridge
  roscpp
  rospy
  std_msgs
  message_generation
  sensor_msgs
  image_transport
)

find_package(OpenCV REQUIRED)

add_message_files(
  FILES
  can.msg
  multican.msg
  Range.msg
  lane.msg
)

generate_messages(
  DEPENDENCIES
  std_msgs
)

catkin_package(
#  INCLUDE_DIRS include
  LIBRARIES pwm_control
  CATKIN_DEPENDS roscpp rospy std_msgs 
)

include_directories(
#  include
  ${catkin_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS}
)

add_executable(pwm_control_node src/pwm_control_node.cpp)
target_link_libraries(pwm_control_node ${catkin_LIBRARIES})
add_dependencies(pwm_control_node pwm_control_generate_messages_cpp)

add_executable(lane_detect_node src/lane_detect_node.cpp)
target_link_libraries(lane_detect_node ${catkin_LIBRARIES})
target_link_libraries(lane_detect_node ${OpenCV_LIBRARIES})
add_dependencies(lane_detect_node pwm_control_generate_messages_cpp)

add_executable(control_tower_node src/control_tower_node.cpp)
target_link_libraries(control_tower_node ${catkin_LIBRARIES})
target_link_libraries(control_tower_node ${OpenCV_LIBRARIES})
add_dependencies(control_tower_node pwm_control_generate_messages_cpp)

#message("catkin_LIBRARIES", ${catkin_LIBRARIES})
message("OpenCV_LIBRARIES", ${OpenCV_LIBRARIES})

