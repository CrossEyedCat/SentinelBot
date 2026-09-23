"""Record the overview camera for the whole mission and remember when each frame happened.

    python3 tools/mission_grab.py <out.mp4> <fps_out> <max_wall_seconds>

The simulation does not run at real time with nine ranging sensors on, so frames are timed by the
simulation clock they carry, not by when they turn up here. The camera is configured for a few
frames per simulation second and the clip is written at fps_out, which makes the video a fixed
speed-up of the run: fps_out / camera_rate. Each frame's simulation stamp is written to
<out.mp4>.stamps so the costmap panel can be rendered one frame per camera frame and stay in sync.

Stops on MISSION_COMPLETE (plus a short tail) or max_wall_seconds, whichever comes first.
"""
import subprocess
import sys
import time

import rclpy
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Image
from std_msgs.msg import String

W, H = 960, 720
TAIL = 4.0      # s of wall time to keep recording after MISSION_COMPLETE

out, fps_out, max_wall = sys.argv[1], float(sys.argv[2]), float(sys.argv[3])
stamps = open(out + ".stamps", "w")

ff = subprocess.Popen(
    ["ffmpeg", "-y", "-loglevel", "error",
     "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", "%dx%d" % (W, H), "-framerate", "%.4f" % fps_out,
     "-i", "-", "-c:v", "libx264", "-preset", "ultrafast", "-crf", "20", "-pix_fmt", "yuv420p", out],
    stdin=subprocess.PIPE,
)

rclpy.init()
node = Node("mission_grab")
state = {"n": 0, "stop_at": None, "last": -1.0, "first": None, "dropped": 0}


def cb(msg):
    if msg.width != W or msg.height != H or msg.encoding != "rgb8":
        return
    t = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
    if t <= state["last"]:
        # a frame stamped on another clock - a leftover simulator on the same ROS domain will do
        # this - must never enter the timeline the costmap panel is rendered against
        state["dropped"] += 1
        return
    state["last"] = t
    if state["first"] is None:
        state["first"] = t
    ff.stdin.write(bytes(msg.data))
    stamps.write("%.6f\n" % t)
    state["n"] += 1


def mission_cb(msg):
    if ("MISSION_COMPLETE" in msg.data or "MISSION_ABORTED" in msg.data) and state["stop_at"] is None:
        state["stop_at"] = time.time() + TAIL
        node.get_logger().info("[GRAB] mission complete, closing the clip")


qos = QoSProfile(reliability=ReliabilityPolicy.RELIABLE, durability=DurabilityPolicy.VOLATILE,
                 history=HistoryPolicy.KEEP_LAST, depth=30)
node.create_subscription(Image, "/overview_cam", cb, qos)
node.create_subscription(String, "/mission_status", mission_cb, 10)

t0 = time.time()
while time.time() - t0 < max_wall:
    rclpy.spin_once(node, timeout_sec=0.01)
    if state["stop_at"] is not None and time.time() > state["stop_at"]:
        break
node.destroy_node()
rclpy.try_shutdown()
ff.stdin.close()
ff.wait()
stamps.close()
span = state["last"] - (state["first"] or state["last"])
rate = (state["n"] - 1) / span if state["n"] > 1 and span > 0 else 0.0
open(out + ".rate", "w").write("%.4f\n" % rate)
print("   camera: %d frames over %.0f s of simulation (%.2f fps sim, %d dropped) "
      "-> %.1f s of video, x%.1f real time"
      % (state["n"], span, rate, state["dropped"], state["n"] / fps_out, fps_out / rate if rate else 0.0))
