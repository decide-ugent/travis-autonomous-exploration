FROM ros:jazzy

# Software rendering — no GPU required (uses Mesa llvmpipe)
ENV LIBGL_ALWAYS_SOFTWARE=1
ENV MESA_GL_VERSION_OVERRIDE=3.3

# Make sure everything is up to date before building from source
RUN apt-get update \
  && apt-get upgrade -y \
  && apt-get -y install python3-pip \
  && apt-get clean

RUN pip install pandas --break-system-packages

RUN apt-get update && apt-get install -q -y --no-install-recommends \
    # python3-colcon-ros \
    ros-jazzy-realtime-tools \
    ros-jazzy-test-msgs \
    ros-jazzy-navigation2 \
    ros-jazzy-nav2-bringup \
    ros-jazzy-rviz2 \
    ros-jazzy-rqt \
    ros-jazzy-topic-tools \
    ros-jazzy-teleop-twist-keyboard \
    ros-jazzy-rqt-common-plugins \
    ros-jazzy-image-transport-plugins \
    ros-jazzy-rmw-cyclonedds-cpp \
    python3-networkx\
    # ros-jazzy-gazebo* \
    # X11 and Mesa software rendering
    libgl1-mesa-dri \
    mesa-utils \
    x11-apps \
    && apt-get clean
# Source - https://stackoverflow.com/a/66502363
# Posted by jubnzv
# Retrieved 2026-04-10, License - CC BY-SA 4.0

#ros-jazzy-topic-tools is usefull for cross ros2 distro (humble-> jazzy for instance)

# RUN apt-get update && apt-get install -y software-properties-common \
#     && add-apt-repository ppa:openrobotics/gazebo11-gz-cli \
#     && apt-get update \
#     && sudo apt-get install gazebo11 -y --no-install-recommends

RUN apt-get install -y sudo \
  && echo "ubuntu ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers \
  && apt-get clean

RUN mkdir -p /ros2_ws/src \
  && chown -R ubuntu:ubuntu /ros2_ws
WORKDIR /ros2_ws

RUN echo "source /opt/ros/jazzy/setup.bash" >> /home/ubuntu/.bashrc
RUN echo "source /ros2_ws/install/setup.bash" >> /home/ubuntu/.bashrc
