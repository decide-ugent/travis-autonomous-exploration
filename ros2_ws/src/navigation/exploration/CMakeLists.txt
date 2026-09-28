cmake_minimum_required(VERSION 3.8)
project(exploration)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)

# Install Python package so `from exploration.xxx import ...` works after build
ament_python_install_package(${PROJECT_NAME})

# Install the ROS2 node scripts as ros2-run executables
# (recorder.py and manual_exploration.py are the system-test layer: run
# recorder alongside any run; manual_exploration replaces the exploration
# node for human-driven baseline runs).
install(PROGRAMS
  exploration/ros2_exploration_node.py
  tests/system/recorder.py
  tests/system/manual_exploration.py
  DESTINATION lib/${PROJECT_NAME}
)

# Install launch and config directories into share/
install(DIRECTORY launch config rviz
  DESTINATION share/${PROJECT_NAME}
)

# Scene config+baseline YAMLs for the system tests: installed so the recorder
# can resolve a bare 'baselines/baseline_<scene>.yaml' from any cwd. NOTE:
# these are a build-time COPY — evaluate_run.py --save-baseline should target
# the SOURCE tree file (tests/system/baselines/) so the baseline is versioned.
install(DIRECTORY tests/system/baselines
  DESTINATION share/${PROJECT_NAME}
)

install(DIRECTORY rviz DESTINATION share/${PROJECT_NAME})

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  # the following line skips the linter which checks for copyrights
  # comment the line when a copyright and license is added to all source files
  set(ament_cmake_copyright_FOUND TRUE)
  # the following line skips cpplint (only works in a git repo)
  # comment the line when this package is in a git repo and when
  # a copyright and license is added to all source files
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()
