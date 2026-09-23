#!/usr/bin/env bash
# Mission video: the robot drives a full lap around every column, filmed from an overview camera
# beside the arena, with the planner's costmap rendered next to it.
#
#   tools/record_column_tour.sh <out.mp4> <max_wall_seconds> [extra column_tour ros args...]
#
# The simulation does not run at real time with nine ranging sensors on, so the camera is set to a
# few frames per SIMULATION second and the clip is written at SPEEDUP times that rate: the video is
# a fixed speed-up of the run, and every frame carries the simulation time that the costmap panel
# is rendered against. Needs ffmpeg. (no `set -u`: ROS setup.bash references unset variables)
OUT="${1:?usage: record_column_tour.sh <out.mp4> <max_wall_seconds> [extra args]}"
MAX_WALL="${2:-2400}"
HERE="$(cd "$(dirname "$0")" && pwd)"
CAM_RATE=3.5          # frames per simulation second, matches tools/overview_cam.sdf
SPEEDUP=6
FPS=21                # CAM_RATE * SPEEDUP
# MODE=waypoints  a point goal every 0.34 m, each one an arrival (the default)
# MODE=continuous the lap published as one path, driven by the hand-written carrot law
# MODE=pilot      the same path, steered by the imitation-trained policy
MODE="${MODE:-waypoints}"
# MAP=<name>     one of the generated arenas in worlds/ (tools/make_worlds.py); unset = turtlebot3_world
MAP="${MAP:-}"
case $MODE in
  waypoints)  CONT=false; PILOT=false; TIMEOUT=90.0 ;;   # a double: the node declares it as one
  continuous) CONT=true;  PILOT=false; TIMEOUT=240.0 ;;
  pilot)      CONT=true;  PILOT=true;  TIMEOUT=240.0 ;;
  *) echo "unknown MODE $MODE (waypoints|continuous|pilot)"; exit 2 ;;
esac
source /opt/ros/jazzy/setup.bash
source "${ROS_WS:-$HOME/ros2_ws}/install/setup.bash"   # ROS_WS: your colcon workspace
export ROS_DOMAIN_ID=${ROS_DOMAIN_ID:-7}

# Every pattern here has to be one this script's own command line cannot match. A plain
# "pkill -f column_tour" matches the path of this file, so the script killed itself on the
# first cleanup line and left the simulator running; the next run then had two servers
# publishing /odom and /scan into the same ROS domain, which is unrecoverable nonsense.
cleanup(){
  # the node's command line is .../lib/sentinel_patrol/column_tour --ros-args: "sentinel_patrol column_tour"
  # never matched it, and two missions from earlier runs were found still spinning hours later
  pkill -f "lib/sentinel_patrol/column_tour"; pkill -f "mission_logger.py"; pkill -f "mission_grab.py"
  pkill -f patrol_fsm; pkill -f occupancy_mapper; pkill -f path_planner; pkill -f scan_to_range
  pkill -f parameter_bridge; pkill -f mecanum_base_sim; pkill -f "gz sim"
  sleep 2; pkill -9 -f "gz sim"
}
trap cleanup EXIT
# Two recorders on one ROS domain means two FSMs driving one robot. A run whose simulator was
# killed from outside kept its recorder alive, and it attached a second stack to the next run.
# anchored at the start of the command line: a wrapper that merely mentions this script in a
# `bash -c "..."` string is not a second recorder, and counting it stopped a run before it began
if [ "$(pgrep -fc '^bash [^ ]*record_column_tour.sh')" -gt 1 ]; then echo "!!! another recorder is running"; pgrep -af record_column_tour.sh; exit 1; fi
cleanup                     # a leftover simulator from an earlier run would poison this one
sleep 1

PKG=$(ros2 pkg prefix --share sentinel_patrol)
TB3=$(ros2 pkg prefix --share turtlebot3_gazebo)
export GZ_SIM_RESOURCE_PATH=$PKG/models:$TB3/models
WORLD_ARGS=""; MISSION_ARGS=""
if [ -n "$MAP" ]; then
  [ -f "$PKG/worlds/$MAP.map" ] || { echo "no such map: $PKG/worlds/$MAP.map"; exit 2; }
  source "$PKG/worlds/$MAP.map"
  WORLD_ARGS="world:=$PKG/worlds/$WORLD x_pose:=$SPAWN_X y_pose:=$SPAWN_Y"
  MISSION_ARGS="-p spawn_x:=$SPAWN_X -p spawn_y:=$SPAWN_Y -p ring_radius:=$RING -p transit_via_planner:=$TRANSIT -p columns:=$COLUMNS"
  echo "=== map $MAP: $WORLD, spawn ($SPAWN_X, $SPAWN_Y), ring $RING m, transit via planner: $TRANSIT, planner margin ${PLANNER_MARGIN:-default}"
fi
W="$(mktemp -d)"; LOG="$W/log"; mkdir -p "$LOG"
export PILOT_TRACE="$LOG/pilot_trace.csv"      # what the learned pilot saw and did, step by step
echo "=== workdir $W"

ros2 launch sentinel_patrol sentinel_gazebo.launch.py gui:=false $WORLD_ARGS > "$W/gazebo.log" 2>&1 &
for i in $(seq 1 90); do timeout 3 ros2 topic echo /odom --once >/dev/null 2>&1 && break; sleep 1; done
sleep 3
ros2 run ros_gz_sim create -file "$HERE/overview_cam.sdf" -name overview_cam > "$W/spawn_cam.log" 2>&1
ros2 run ros_gz_bridge parameter_bridge "/overview_cam@sensor_msgs/msg/Image[gz.msgs.Image" --ros-args -p use_sim_time:=true > "$W/bridge_cam.log" 2>&1 &

ros2 launch sentinel_patrol patrol.launch.py holonomic:=true sensor_mode:=per_sensor map:=true plan:=true rviz:=false use_sim_time:=true pilot:=$PILOT > "$W/patrol.log" 2>&1 &
# One simulator is two matching processes - the /bin/sh -c ruby wrapper and the server it
# execs - so only the server itself is counted, anchored at the start of the command line.
N=$(pgrep -fc "^gz sim -r -s" || echo 0)
if [ "$N" != "1" ]; then echo "!!! $N gz sim servers are running, expected 1"; pgrep -af "gz sim"; exit 1; fi
echo "=== waiting for the costmap and the camera"
for i in $(seq 1 90); do timeout 3 ros2 topic echo /costmap --once >/dev/null 2>&1 && break; sleep 1; done
for i in $(seq 1 40); do timeout 4 ros2 topic echo /overview_cam --once --field encoding 2>/dev/null | grep -q rgb8 && break; sleep 1; done
sleep 2

# Mission profile. Circling a column means living inside the distances the reactive layer exists
# to protect: on a 0.55 m ring the column is 0.40 m from the chassis and 0.22 m from a corner,
# against a patrol d_stop of 0.50 m and an ir_critical of 0.12 m. With the shipped profile the
# robot reads its own lap as a continuous emergency - a first attempt spent 26 back-offs and 18
# stall recoveries on it and travelled 69 m to walk 31 m of laps. The thresholds are lowered to
# match a manoeuvre whose clearance is guaranteed by geometry rather than by sensing, the same
# trade the final approach already makes (d_stop_final 0.22) to reach a goal against a wall.
# goal_yaw_align is raised so the robot arcs through the ring instead of stopping to turn at every
# waypoint. Set live on the running node; the shipped config/patrol_params.yaml is left alone.
for kv in "d_stop 0.20" "d_stop_final 0.15" "d_clear 0.35" "ir_critical 0.07" "ir_steer 0.15" "goal_yaw_align 0.75" "v_goal 0.30" "v_final 0.15" "d_final 0.20"; do
  ros2 param set /patrol_fsm $kv > /dev/null || echo "  could not set $kv"
done
echo "=== mission profile: d_stop $(ros2 param get /patrol_fsm d_stop) ir_critical $(ros2 param get /patrol_fsm ir_critical)"
# the generated arenas carry a wider planner margin: a route past a wall end at the default
# 0.05 m put the robot against the end, where it stayed until the mission gave up
[ -n "$PLANNER_MARGIN" ] && { ros2 param set /path_planner safety_margin $PLANNER_MARGIN > /dev/null || echo "  could not set the planner margin"; }

python3 "$HERE/mission_logger.py" "$LOG" 1800 > "$W/logger.log" 2>&1 &
LOGPID=$!
python3 "$HERE/mission_grab.py" "$W/cam.mp4" $FPS $MAX_WALL > "$W/grab.log" 2>&1 &
GRABPID=$!
sleep 2
ros2 run sentinel_patrol column_tour --ros-args -p use_sim_time:=true -p continuous:=$CONT -p goal_timeout:=$TIMEOUT $MISSION_ARGS "${@:3}" > "$W/mission.log" 2>&1 &
echo "=== mission running in $MODE mode, recording up to $MAX_WALL s of wall time"
wait $GRABPID
cat "$W/grab.log"
# the logger closes itself a few seconds after MISSION_COMPLETE; give it that time before
# signalling, because a signal mid-write is how the first full run lost its log
for i in $(seq 1 25); do kill -0 $LOGPID 2>/dev/null || break; sleep 1; done
kill -INT $LOGPID 2>/dev/null
wait $LOGPID 2>/dev/null
cat "$W/logger.log"
echo "=== mission log tail:"; grep MISSION "$W/mission.log" | tail -6

python3 "$HERE/mission_panel.py" "$LOG" "$W/cam.mp4.stamps" "$W/panel.mp4" $FPS
RATE=$(cat "$W/cam.mp4.rate" 2>/dev/null || echo "$CAM_RATE")
REAL=$(python3 -c "print('%.1f' % ($FPS / float('$RATE')))")
LABEL="$MODE"
[ -n "$MAP" ] && LABEL="$MAP - $LABEL"
printf "one lap around every column - %s - x%s real time" "$LABEL" "$REAL" > "$W/label.txt"
FONT=/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf
ffmpeg -y -loglevel error -i "$W/cam.mp4" -i "$W/panel.mp4" -filter_complex "[0:v]drawbox=x=0:y=0:w=iw:h=48:color=black@0.55:t=fill,drawtext=fontfile=$FONT:textfile=$W/label.txt:fontcolor=white:fontsize=21:x=16:y=13[a];[a][1:v]hstack=inputs=2[v]" -map "[v]" -c:v libx264 -preset veryfast -crf 21 -pix_fmt yuv420p "$OUT"
cp "$W/cam.mp4.stamps" "$LOG/cam.stamps" 2>/dev/null   # the renderer needs the frame clock
rm -rf "$(dirname "$OUT")/mission_log"; cp -r "$LOG" "$(dirname "$OUT")/mission_log"
ls -la "$OUT"
ffprobe -v error -show_entries format=duration -show_entries stream=width,height -of default=nw=1 "$OUT"
