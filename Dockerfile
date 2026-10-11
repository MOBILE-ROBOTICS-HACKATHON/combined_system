# Self-Contained Dockerfile for Combined Perception & Robot Drive System
FROM ros:humble-ros-base

ENV DEBIAN_FRONTEND=noninteractive
ENV RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
ENV ROS_DOMAIN_ID=0

# 1. System packages & ROS 2 dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    git \
    curl \
    iputils-ping \
    net-tools \
    python3-pip \
    python3-colcon-common-extensions \
    ros-humble-rmw-cyclonedds-cpp \
    ros-humble-geometry-msgs \
    ros-humble-std-msgs \
    ros-humble-sensor-msgs \
    ros-humble-nav-msgs \
    ros-humble-tf2-ros \
    ros-humble-cv-bridge \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# 2. Python ML, CV, Hardware & Utility dependencies
RUN pip3 install --default-timeout=1000 --retries=20 --ignore-installed \
    "numpy<2" \
    "scipy" \
    "opencv-python-headless<4.11" \
    "pyyaml" \
    "websockets" \
    "cflib" \
    "matplotlib" \
    "psutil" \
    "tqdm" \
    "py-cpuinfo" \
    "requests"
RUN pip3 install --default-timeout=1000 --retries=20 --ignore-installed torch torchvision --index-url https://download.pytorch.org/whl/cpu
RUN pip3 install --default-timeout=1000 --retries=20 --no-deps ultralytics

# 3. Setup workspace directory
WORKDIR /combined_ws
RUN mkdir -p /combined_ws/src

# 4. Copy self-contained components
COPY ws/src/team_pkg /combined_ws/src/team_pkg
COPY RescueRobotBrain /combined_ws/RescueRobotBrain
COPY brain_node.py /combined_ws/brain_node.py

# 5. Build ROS 2 workspace
SHELL ["/bin/bash", "-c"]
RUN source /opt/ros/humble/setup.bash && \
    cd /combined_ws && \
    colcon build --symlink-install

# 6. Source scripts in bashrc
RUN echo "source /opt/ros/humble/setup.bash" >> ~/.bashrc && \
    echo "source /combined_ws/install/setup.bash" >> ~/.bashrc

WORKDIR /combined_ws
CMD ["bash"]
