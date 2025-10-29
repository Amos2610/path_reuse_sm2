cmake_minimum_required(VERSION 3.8)
project(path_reuse_sm2_interfaces)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)
find_package(rosidl_default_generators REQUIRED)

find_package(std_msgs REQUIRED)
find_package(trajectory_msgs REQUIRED)
find_package(ament_lint_auto REQUIRED)

rosidl_generate_interfaces(${PROJECT_NAME}
  "srv/SetPathSeedTrajectory.srv"
  "srv/GetPathSeedTrajectory.srv"
  "srv/DecodePathSeed.srv"
  "srv/EncodePathSeed.srv"
  "msg/PathSeed.msg"
  DEPENDENCIES std_msgs trajectory_msgs
)

ament_export_dependencies(rosidl_default_runtime)
ament_package()

if(BUILD_TESTING)
  set(ament_cmake_copyright_FOUND TRUE)
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()
