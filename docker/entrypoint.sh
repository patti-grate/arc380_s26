#!/bin/bash
set -e

# Source ROS 2 and workspace
source /opt/ros/jazzy/setup.bash
source /ros2_ws/install/setup.bash

# If arguments are passed, run them directly (e.g., docker exec / custom commands)
if [ "$#" -gt 0 ]; then
    exec "$@"
fi

# ── 1. Virtual display ────────────────────────────────────────────────────────
echo "[1/5] Starting virtual display (Xvfb)..."
Xvfb :1 -screen 0 1920x1080x24 -ac +extension GLX &
sleep 2
export DISPLAY=:1

# ── 2. Window manager ─────────────────────────────────────────────────────────
echo "[2/5] Starting window manager (openbox)..."
openbox &
sleep 1

# ── 3. VNC server ─────────────────────────────────────────────────────────────
echo "[3/5] Starting VNC server (x11vnc)..."
x11vnc -display :1 -nopw -forever -shared -bg -quiet
sleep 1

# ── 4. noVNC web interface ────────────────────────────────────────────────────
echo "[4/5] Starting noVNC web interface on port 6080..."
websockify --web /usr/share/novnc/ --daemon 6080 localhost:5900

echo ""
echo "========================================================================"
echo "  GUI available at:  http://localhost:6080/vnc.html"
echo "  VNC direct:        localhost:5900  (no password)"
echo "========================================================================"
echo ""

# ── 5. Launch simulation ──────────────────────────────────────────────────────
echo "[5/5] Launching MoveIt + Gazebo simulation..."
ros2 launch abb_irb120_gazebo gz_moveit.launch.py &
SIM_PID=$!

# Wait for the Gazebo server to be ready before connecting the GUI
echo "Waiting for Gazebo server to initialise (15 s)..."
sleep 15

# Connect the Gazebo GUI client to the already-running server
echo "Launching Gazebo GUI client..."
gz sim -g &

echo ""
echo "All processes started."
echo "  • Open http://localhost:6080/vnc.html in your browser to see the GUI."
echo "  • To run Python scripts from the container:"
echo "      docker exec -it arc380_ros2 bash"
echo ""

# Keep container alive; exit if the main simulation process dies
wait $SIM_PID
