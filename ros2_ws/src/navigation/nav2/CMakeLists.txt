cmake_minimum_required(VERSION 3.8)
project(nav2)

find_package(ament_cmake REQUIRED)

install(DIRECTORY launch config
  DESTINATION share/${PROJECT_NAME}
)

# PROGRAMS (not FILES) so the executable bit survives the install, which is what makes `ros2 run nav2 save_map.sh` work.
install(PROGRAMS scripts/save_map.sh
  DESTINATION lib/${PROJECT_NAME}
)

ament_package()
