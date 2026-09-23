#!/usr/bin/env bash
# End-to-end demo: Gazebo (Harmonic, ROS2 Jazzy) + TurtleBot3 + sentinel_patrol FSM.
#
#   tools/run_demo.sh [patrol_seconds] [extra launch args...]
#
# Timeline:
#   0 s      Gazebo turtlebot3_world, GUI camera follows the robot
#   ~10 s    FSM + /scan->Range adapter, operator "start"
#   ...      free patrol: PATROLLING <-> OBSTACLE_AVOIDANCE (screenshots on every state change)
#   +T       tipping event: the robot is set on its side with /world/default/set_pose -> ALERT
#   +T+8 s   operator "reset" -> IDLE
#
# Output: ~/sentinel_demo/<timestamp>/  gazebo.log patrol.log states.log alerts.log
#         transitions.log timeline.csv shot_NN_<STATE>.png
# (no `set -u`: ROS setup.bash references unset variables)
PATROL_SECONDS="${1:-90}"
shift 2>/dev/null || true
EXTRA_ARGS="$*"
# ROBOT=tb3  -> stock TurtleBot3 Burger (turtlebot3_gazebo), /cmd_vel is TwistStamped on Jazzy
# ROBOT=mk2  -> SentinelBot Mk II (our SDF: mecanum/holonomic base, LiDAR + 4 US + 4 IR + IMU)
ROBOT="${ROBOT:-mk2}"
if [ "$ROBOT" = "mk2" ]; then
  GZ_LAUNCH="sentinel_patrol sentinel_gazebo.launch.py"
  MODEL_NAME="sentinel_mk2"
  case " $EXTRA_ARGS " in
    *" cmd_vel_stamped:=true "*)
      echo "ERROR: the Mk II base controller subscribes to geometry_msgs/Twist on /cmd_vel;" >&2
      echo "       cmd_vel_stamped:=true would leave the robot motionless. Use ROBOT=tb3 for TwistStamped." >&2
      exit 2 ;;
  esac
  EXTRA_ARGS="holonomic:=true sensor_mode:=per_sensor $EXTRA_ARGS"
  TIP_POSE='name: "sentinel_mk2", position: {x: -2.0, y: -0.5, z: 0.20}, orientation: {x: 0.643, y: 0.0, z: 0.0, w: 0.766}'
else
  GZ_LAUNCH="turtlebot3_gazebo turtlebot3_world.launch.py"
  MODEL_NAME="burger"
  EXTRA_ARGS="cmd_vel_stamped:=true $EXTRA_ARGS"
  TIP_POSE='name: "burger", position: {x: -2.0, y: -0.5, z: 0.15}, orientation: {x: 0.42, y: 0.0, z: 0.0, w: 0.91}'
fi
echo "robot=$ROBOT  launch args: $EXTRA_ARGS"

source /opt/ros/jazzy/setup.bash
source "${ROS_WS:-$HOME/ros2_ws}/install/setup.bash"   # ROS_WS: your colcon workspace
export TURTLEBOT3_MODEL=burger
export ROS_DOMAIN_ID=${ROS_DOMAIN_ID:-7}
PKG="$(cd "$(dirname "$0")/.." && pwd)"

OUT="$HOME/sentinel_demo/$(date +%Y%m%d_%H%M%S)"
mkdir -p "$OUT"
echo "logs -> $OUT"

cleanup() {
  echo "--- stopping"
  pkill -f "state_watcher" 2>/dev/null
  pkill -f "patrol_fsm" 2>/dev/null
  pkill -f "scan_to_range" 2>/dev/null
  pkill -f "ros2 topic echo" 2>/dev/null
  pkill -f "turtlebot3_world" 2>/dev/null
  pkill -f "sentinel_gazebo" 2>/dev/null
  pkill -f "parameter_bridge" 2>/dev/null
  pkill -f "gz sim" 2>/dev/null
  pkill -f "ros_gz_bridge" 2>/dev/null
  sleep 2
  pkill -9 -f "gz sim" 2>/dev/null
}
trap cleanup EXIT

gzsvc() {  # gzsvc <service> <reqtype> <request>
  gz service -s "$1" --reqtype "$2" --reptype gz.msgs.Boolean --timeout 3000 --req "$3" > /dev/null 2>&1
}

# 1. Gazebo
ros2 launch $GZ_LAUNCH > "$OUT/gazebo.log" 2>&1 &
echo "--- waiting for /scan"
for i in $(seq 1 90); do
  if timeout 3 ros2 topic echo /scan --once > /dev/null 2>&1; then echo "scan ok after ${i}s"; break; fi
  sleep 1
done
timeout 3 ros2 topic echo /imu --once > /dev/null 2>&1 && echo "imu ok" || echo "WARNING: no /imu"

# camera follows the robot (CameraTracking GUI plugin)
sleep 3
follow_robot() {
  gzsvc /gui/follow gz.msgs.StringMsg "data: \"$MODEL_NAME\""
  gzsvc /gui/follow/offset gz.msgs.Vector3d 'x: -1.8, y: 0.0, z: 1.6'
}
# fixed world-frame camera for the tipped robot at the spawn point (-2.0, -0.5): 1 m in front,
# 1 m up, looking back down at it (yaw pi, pitch 0.675 rad -> quaternion x=-0.331 z=0.944)
camera_on_spawn() {
  gzsvc /gui/follow gz.msgs.StringMsg 'data: ""'
  gzsvc /gui/move_to/pose gz.msgs.GUICamera \
    'pose: {position: {x: -1.0, y: -0.5, z: 1.0}, orientation: {x: -0.331, y: 0.0, z: 0.944, w: 0.0}}'
}
follow_robot

# 2. FSM + adapter + recorders
ros2 launch sentinel_patrol patrol.launch.py $EXTRA_ARGS > "$OUT/patrol.log" 2>&1 &
sleep 4
python3 "$PKG/tools/state_watcher.py" "$OUT" > "$OUT/watcher.log" 2>&1 &
ros2 topic echo /patrol_state std_msgs/msg/String > "$OUT/states.log" 2>&1 &
ros2 topic echo /robot_alert std_msgs/msg/String > "$OUT/alerts.log" 2>&1 &
sleep 2

# 3. Operator start, free patrol
ros2 topic pub --once /patrol_cmd std_msgs/msg/String "{data: start}" > /dev/null 2>&1
echo "--- patrol started, free run ${PATROL_SECONDS}s"
sleep "$PATROL_SECONDS"

# 4. Tipping event: put the robot on its side at the spawn point (roll ~ 50 deg, 15 cm up)
echo "--- injecting tipping event via /world/default/set_pose"
gzsvc /world/default/set_pose gz.msgs.Pose "$TIP_POSE"
sleep 0.5
camera_on_spawn       # the follow camera would inherit the tipped robot's roll; use a fixed view instead
sleep 7.5

# 5. Operator reset
ros2 topic pub --once /patrol_cmd std_msgs/msg/String "{data: reset}" > /dev/null 2>&1
echo "--- reset sent"
sleep 5

# 6. Summary
grep -E "\[FSM\]|ALERT reason" "$OUT/patrol.log" | sed -E 's/^\[patrol_fsm-[0-9]+\] //' > "$OUT/transitions.log"
echo "=== transitions ==="
cat "$OUT/transitions.log"
echo "=== state histogram (10 Hz ticks) ==="
grep -E "^data:" "$OUT/states.log" | sort | uniq -c
echo "=== alerts ==="
grep -E "^data:" "$OUT/alerts.log" || echo "(none)"
echo "=== screenshots ==="
ls -1 "$OUT"/shot_*.png 2>/dev/null || echo "(none)"
