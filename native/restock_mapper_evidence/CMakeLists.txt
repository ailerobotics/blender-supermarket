cmake_minimum_required(VERSION 3.16)
project(restock_mapper_evidence LANGUAGES C CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
option(EVIDENCE_BUILD_ROS_INTERFACES "Generate typed interfaces (no producer node)" ON)
option(EVIDENCE_BUILD_PRODUCER "Build the native evidence producer component (needs installed nvblox_ros)" OFF)
enable_testing()

# Pure, ROS-independent admission/ledger policy test. Never constructs a node.
add_executable(evidence_state_test test/state_test.cpp)
target_include_directories(evidence_state_test PRIVATE include)
target_compile_options(evidence_state_test PRIVATE -Wall -Wextra -Werror -UNDEBUG)
add_test(NAME evidence_state COMMAND evidence_state_test)

if(EVIDENCE_BUILD_ROS_INTERFACES)
  find_package(ament_cmake REQUIRED)
  find_package(rosidl_default_generators REQUIRED)
  find_package(builtin_interfaces REQUIRED)
  find_package(nvblox_msgs REQUIRED)
  rosidl_generate_interfaces(${PROJECT_NAME}
    "msg/MapIdentity.msg" "msg/Acquisition.msg" "msg/LayerCount.msg"
    "msg/SensorBinding.msg" "msg/EvidenceSnapshot.msg"
    "srv/EmptyGeneration.srv" "srv/OpenAdmission.srv" "srv/RegisterAcquisition.srv"
    DEPENDENCIES builtin_interfaces nvblox_msgs)
  ament_export_dependencies(rosidl_default_runtime)

  # The producer subclass links the installed nvblox_ros/nvblox libraries. It is
  # only compiled/linked here: building the component proves the overridden
  # virtual signatures and symbol/ABI closure against the installed library.
  # The node is never constructed by this build (its constructor may allocate GPU).
  if(EVIDENCE_BUILD_PRODUCER)
    find_package(rclcpp REQUIRED)
    find_package(rclcpp_components REQUIRED)
    find_package(sensor_msgs REQUIRED)
    # nvblox_ros's exported targets name these imported targets in their link
    # interface but its export-dependencies extras does not find them, so they
    # must exist before nvblox_rosConfig loads its Targets file.
    find_package(rosbag2_cpp REQUIRED)
    find_package(gflags QUIET)
    find_package(glog REQUIRED)
    find_package(nvblox_ros REQUIRED)
    add_library(evidence_producer SHARED src/producer_node.cpp)
    target_include_directories(evidence_producer PUBLIC
      "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>"
      "$<INSTALL_INTERFACE:include>")
    ament_target_dependencies(evidence_producer
      rclcpp rclcpp_components nvblox_ros nvblox_msgs sensor_msgs builtin_interfaces)
    rosidl_get_typesupport_target(evidence_cpp_typesupport ${PROJECT_NAME} rosidl_typesupport_cpp)
    target_link_libraries(evidence_producer ${evidence_cpp_typesupport})
    rclcpp_components_register_nodes(evidence_producer "restock_mapper_evidence::EvidenceProducerNode")
    install(TARGETS evidence_producer
      ARCHIVE DESTINATION lib LIBRARY DESTINATION lib RUNTIME DESTINATION bin)
    install(DIRECTORY include/ DESTINATION include)
    ament_export_libraries(evidence_producer)
  endif()

  ament_package()
endif()
