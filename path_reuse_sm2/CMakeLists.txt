cmake_minimum_required(VERSION 3.10)
project(path_reuse_sm2)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED) 
find_package(rclcpp REQUIRED)
find_package(rclpy REQUIRED)
find_package(std_msgs REQUIRED)
find_package(trajectory_msgs REQUIRED)
find_package(yasmin REQUIRED)
find_package(yasmin_viewer REQUIRED)
find_package(ament_cmake_python REQUIRED)

# Pythonパッケージのインストール（CMakeListsから見て、直下の path_reuse_sm2/ がパッケージ）
ament_python_install_package(${PROJECT_NAME}
  PACKAGE_DIR path_reuse_sm2
)

# Python パッケージをインストール
install(PROGRAMS
  path_reuse_sm2/sm_node.py
  DESTINATION lib/${PROJECT_NAME}
)
install(DIRECTORY ${PROJECT_NAME}
  DESTINATION ${PYTHON_INSTALL_DIR}
)

# 共有データ（launch/io など）があれば
install(DIRECTORY launch io
  DESTINATION share/${PROJECT_NAME}
  USE_SOURCE_PERMISSIONS
)
ament_package()
