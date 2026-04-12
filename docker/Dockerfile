# syntax=docker/dockerfile:1

FROM ubuntu:24.04 AS base

# Avoid interactive prompts during package installation
ARG DEBIAN_FRONTEND=noninteractive
ENV LANG=en_US.UTF-8
ENV LC_ALL=en_US.UTF-8

# ── Locale ────────────────────────────────────────────────────────────────────
RUN apt-get update && apt-get install -y locales && \
    locale-gen en_US en_US.UTF-8 && \
    update-locale LC_ALL=en_US.UTF-8 LANG=en_US.UTF-8 && \
    rm -rf /var/lib/apt/lists/*

# ── ROS 2 Jazzy apt repository (Ubuntu 24.04 "noble") ─────────────────────────
RUN apt-get update && apt-get install -y \
        curl gnupg software-properties-common && \
    add-apt-repository universe && \
    curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key | \
        gpg --dearmor -o /usr/share/keyrings/ros-archive-keyring.gpg && \
    echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] \
        http://packages.ros.org/ros2/ubuntu noble main" \
        > /etc/apt/sources.list.d/ros2.list && \
    rm -rf /var/lib/apt/lists/*

# ── OSRF Gazebo Harmonic apt repository ───────────────────────────────────────
RUN curl -sSL https://packages.osrfoundation.org/gazebo.gpg | \
        gpg --dearmor -o /usr/share/keyrings/pkgs-osrf-archive-keyring.gpg && \
    echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/pkgs-osrf-archive-keyring.gpg] \
        http://packages.osrfoundation.org/gazebo/ubuntu-stable noble main" \
        > /etc/apt/sources.list.d/gazebo-stable.list

# ── ROS 2 + simulation + RealSense packages ───────────────────────────────────
RUN apt-get update && apt-get install -y \
    ros-jazzy-desktop-full \
    ros-jazzy-gz-ros2-control \
    ros-jazzy-moveit \
    ros-jazzy-ros2-control \
    ros-jazzy-ros2-controllers \
    ros-jazzy-rmw-cyclonedds-cpp \
    python3-colcon-common-extensions \
    python3-rosdep \
    python3-pip \
    git \
    && rm -rf /var/lib/apt/lists/*

# ── GUI: Xvfb + x11vnc + noVNC (browser-based remote display) ────────────────
RUN apt-get update && apt-get install -y \
    xvfb \
    x11vnc \
    novnc \
    websockify \
    openbox \
    xterm \
    dbus-x11 \
    libgl1-mesa-dri \
    mesa-utils \
    && rm -rf /var/lib/apt/lists/*

# ── rosdep init ───────────────────────────────────────────────────────────────
RUN rosdep init && rosdep update

RUN python3 -m pip install --no-cache-dir --break-system-packages "protobuf==6.31.1"


FROM base AS workspace

# ── Clone and build workspace ─────────────────────────────────────────────────
WORKDIR /ros2_ws
RUN git clone https://github.com/AdelResearchGroup/arc380_s26.git . && \
    /bin/bash -c "\
        source /opt/ros/jazzy/setup.bash && \
        apt-get update && \
        rosdep install --from-paths src --ignore-src -r -y \
            --skip-keys='warehouse_ros_mongo ament_python joint_state_publisher joint_state_publisher_gui' && \
        colcon build"


FROM base AS final

# ── Bring in built workspace only ─────────────────────────────────────────────
WORKDIR /ros2_ws
COPY --from=workspace /ros2_ws /ros2_ws

# ── CycloneDDS config (enables host ↔ container DDS discovery) ────────────────
COPY cyclonedds.xml /cyclonedds.xml

# ── Environment ───────────────────────────────────────────────────────────────
ENV RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
ENV CYCLONEDDS_URI=/cyclonedds.xml
# Software rendering — required without GPU passthrough
ENV LIBGL_ALWAYS_SOFTWARE=1
ENV GALLIUM_DRIVER=llvmpipe
ENV QT_X11_NO_MITSHM=1
ENV DISPLAY=:1
# Allow Gazebo to resolve package:// URIs for robot meshes
ENV GZ_SIM_RESOURCE_PATH=\
/ros2_ws/install/abb_irb120_description/share:\
/ros2_ws/install/abb_irb120_gazebo/share/abb_irb120_gazebo/models

RUN echo 'source /opt/ros/jazzy/setup.bash' >> /etc/bash.bashrc && \
    echo 'source /ros2_ws/install/setup.bash' >> /etc/bash.bashrc

# ── Entrypoint ────────────────────────────────────────────────────────────────
COPY entrypoint.sh /entrypoint.sh
# Strip Windows CRLF line endings in case the file was edited on Windows
RUN sed -i 's/\r$//' /entrypoint.sh && chmod +x /entrypoint.sh

EXPOSE 6080 5900

ENTRYPOINT ["/entrypoint.sh"]
