#!/usr/bin/env bash
# The FSM demonstration video: every state of the patrol FSM and each transition that Section 5.1 of the
# report provokes, in one Gazebo run of the Mk II in the TurtleBot3 world. Left, the overview camera;
# right, a panel drawn from the same run's logs (tools/fsm_demo_panel.py). The scenario is
# tools/fsm_demo.py; it acts only as an operator or the world could.
#
#   tools/record_fsm_demo.sh <out.mp4> [max_wall_seconds]
#
# The camera takes 7 frames per SIMULATION second and the clip is written at that rate, so the film
# runs in simulation real time whatever speed the simulator manages. Needs ffmpeg.
# (no `set -u`: ROS setup.bash references unset variables)
OUT="${1:?usage: record_fsm_demo.sh <out.mp4> [max_wall_seconds]}"
MAX_WALL="${2:-2400}"
HERE="$(cd "$(dirname "$0")" && pwd)"
FPS_CAM=7             # 7 camera frames per simulation second, shown in real time
FPS=21                # the film's frame rate, the same as the mission films of record_column_tour.sh
source /opt/ros/jazzy/setup.bash
source "${ROS_WS:-$HOME/ros2_ws}/install/setup.bash"   # ROS_WS: your colcon workspace
export ROS_DOMAIN_ID=${ROS_DOMAIN_ID:-7}

# patterns chosen so that none of them matches this script's own command line
cleanup(){
  pkill -f "tools/fsm_demo.py"; pkill -f "mission_grab.py"
  pkill -f patrol_fsm; pkill -f occupancy_mapper; pkill -f path_planner; pkill -f scan_to_range
  pkill -f parameter_bridge; pkill -f mecanum_base_sim; pkill -f "gz sim"
  sleep 2; pkill -9 -f "gz sim"
}
trap cleanup EXIT
cleanup
sleep 1

PKG=$(ros2 pkg prefix --share sentinel_patrol)
TB3=$(ros2 pkg prefix --share turtlebot3_gazebo)
export GZ_SIM_RESOURCE_PATH=$PKG/models:$TB3/models
W="$(mktemp -d)"
echo "=== workdir $W"

ros2 launch sentinel_patrol sentinel_gazebo.launch.py gui:=false > "$W/gazebo.log" 2>&1 &
for i in $(seq 1 90); do timeout 3 ros2 topic echo /odom --once >/dev/null 2>&1 && break; sleep 1; done
sleep 3
ros2 run ros_gz_sim create -file "$HERE/fsm_demo_cam.sdf" -name overview_cam > "$W/spawn_cam.log" 2>&1
ros2 run ros_gz_bridge parameter_bridge "/overview_cam@sensor_msgs/msg/Image[gz.msgs.Image" \
  --ros-args -p use_sim_time:=true > "$W/bridge_cam.log" 2>&1 &
ros2 launch sentinel_patrol patrol.launch.py holonomic:=true sensor_mode:=per_sensor map:=true plan:=true \
  rviz:=false use_sim_time:=true pilot:=false > "$W/patrol.log" 2>&1 &
N=$(pgrep -fc "^gz sim -r -s" || echo 0)
if [ "$N" != "1" ]; then echo "!!! $N gz sim servers are running, expected 1"; pgrep -af "gz sim"; exit 1; fi
echo "=== waiting for the costmap and the camera"
for i in $(seq 1 90); do timeout 3 ros2 topic echo /costmap --once >/dev/null 2>&1 && break; sleep 1; done
for i in $(seq 1 40); do timeout 4 ros2 topic echo /overview_cam --once --field encoding 2>/dev/null | grep -q rgb8 && break; sleep 1; done
sleep 2

python3 "$HERE/mission_grab.py" "$W/cam.mp4" $FPS_CAM $MAX_WALL > "$W/grab.log" 2>&1 &
GRABPID=$!
sleep 2
python3 "$HERE/fsm_demo.py" "$W/demo" -2.0 -0.5 2>&1 | tee "$W/demo.log"
wait $GRABPID
cat "$W/grab.log"
python3 "$HERE/fsm_demo_panel.py" "$W/demo" "$W/cam.mp4.stamps" "$W/panel.mp4" $FPS_CAM || exit 1

# ---------------------------------------------------------------- the film: cards around the run
FONT=/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf
FONT_R=/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf
printf "SentinelBot Mk II in Gazebo Harmonic, real time" > "$W/label.txt"
ffmpeg -y -loglevel error -i "$W/cam.mp4" -i "$W/panel.mp4" -filter_complex \
  "[0:v]drawbox=x=0:y=0:w=iw:h=44:color=black@0.55:t=fill,drawtext=fontfile=$FONT:textfile=$W/label.txt:fontcolor=white:fontsize=20:x=14:y=12[a];[a][1:v]hstack=inputs=2,fps=$FPS,setsar=1[v]" \
  -map "[v]" -c:v libx264 -preset veryfast -crf 21 -pix_fmt yuv420p "$W/body.mp4" || exit 1
n=0; list="$W/list.txt"; : > "$list"
card() {   # card <seconds> <big line> [small line] [small line]
  n=$((n + 1)); local f; f=$(printf '%s/card%02d.mp4' "$W" "$n")
  printf '%s' "$2" > "$W/c$n.0"; local vf="drawtext=fontfile=$FONT:textfile=$W/c$n.0:fontcolor=0xE6E9E4:fontsize=42:x=(w-tw)/2:y=250"
  if [ -n "${3:-}" ]; then printf '%s' "$3" > "$W/c$n.1"; vf="$vf,drawtext=fontfile=$FONT_R:textfile=$W/c$n.1:fontcolor=0x98A2AB:fontsize=25:x=(w-tw)/2:y=340"; fi
  if [ -n "${4:-}" ]; then printf '%s' "$4" > "$W/c$n.2"; vf="$vf,drawtext=fontfile=$FONT_R:textfile=$W/c$n.2:fontcolor=0x98A2AB:fontsize=25:x=(w-tw)/2:y=382"; fi
  ffmpeg -y -loglevel error -f lavfi -i "color=c=0x111417:s=1680x720:d=$1:r=$FPS" -vf "$vf" \
    -c:v libx264 -preset veryfast -crf 20 -pix_fmt yuv420p "$f" || exit 1
  echo "file '$f'" >> "$list"
}
card 6 "SentinelBot: the patrol state machine in Gazebo" \
     "All six states, and each transition of Section 5.1 provoked on purpose, in one run." \
     "Left: the overview camera. Right: state, sensors and the FSM's own log from the same run."
echo "file '$W/body.mp4'" >> "$list"
card 5 "Every transition shown is the FSM's own" \
     "The scenario acted only as an operator or the world could: commands, objects," \
     "a pose, a push and a stopped node. It never set a state or a sensor value."
ffmpeg -y -loglevel error -f concat -safe 0 -i "$list" -c:v libx264 -preset medium -crf 21 -pix_fmt yuv420p \
  -movflags +faststart "$OUT" || exit 1
LOGDIR="$(dirname "$OUT")/fsm_demo_log"
rm -rf "$LOGDIR"; mkdir -p "$LOGDIR"
cp "$W/demo/demo.json" "$W/demo.log" "$W/patrol.log" "$W/cam.mp4.stamps" "$LOGDIR/" 2>/dev/null
printf '%s  %s s  %s\n' "$OUT" "$(ffprobe -v error -show_entries format=duration -of csv=p=0 "$OUT" | cut -d. -f1)" \
  "$(du -h "$OUT" | cut -f1)"
