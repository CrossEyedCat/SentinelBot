#!/usr/bin/env python3
"""Demo helper: watch /patrol_state and take a Gazebo GUI screenshot on every state change.

    python3 tools/state_watcher.py <output_dir>

Writes <output_dir>/shot_<n>_<STATE>.png via the Gazebo Sim GUI "Screenshot" plugin service
(/gui/screenshot) and <output_dir>/timeline.csv with "sim_time,state". Used by run_demo.sh to
collect the screenshots required by Part 4 of the assignment without touching the desktop.
"""
import os
import subprocess
import sys
import threading

import rclpy
from rclpy.node import Node
from std_msgs.msg import String

SHOT_DELAY = 1.5   # seconds after the transition, so the robot is visibly in the new state


class StateWatcher(Node):

    def __init__(self, out_dir: str) -> None:
        super().__init__('state_watcher')
        self.out_dir = out_dir
        self.last = None
        self.seq = 0
        self.timeline = open(os.path.join(out_dir, 'timeline.csv'), 'w')
        self.timeline.write('sim_time,state\n')
        self.create_subscription(String, '/patrol_state', self._cb, 10)

    def _cb(self, msg: String) -> None:
        if msg.data == self.last:
            return
        self.last = msg.data
        self.seq += 1
        t = self.get_clock().now().nanoseconds * 1e-9
        self.timeline.write(f'{t:.2f},{msg.data}\n')
        self.timeline.flush()
        threading.Timer(SHOT_DELAY, self._screenshot, args=(self.seq, msg.data)).start()

    def _screenshot(self, seq: int, state: str) -> None:
        """The Screenshot plugin treats the request as a directory and writes <timestamp>.png
        into it, so give every shot its own directory and rename the result afterwards."""
        tmp_dir = os.path.join(self.out_dir, '_shots', f'{seq:02d}_{state}')
        os.makedirs(tmp_dir, exist_ok=True)
        final = os.path.join(self.out_dir, f'shot_{seq:02d}_{state}.png')
        cmd = ['gz', 'service', '-s', '/gui/screenshot', '--reqtype', 'gz.msgs.StringMsg',
               '--reptype', 'gz.msgs.Boolean', '--timeout', '4000', '--req', f'data: "{tmp_dir}"']
        try:
            subprocess.run(cmd, check=False, capture_output=True, timeout=8)
            threading.Timer(1.5, self._collect, args=(tmp_dir, final)).start()
        except Exception as exc:  # demo helper: report, never crash the watcher
            self.get_logger().warning(f'screenshot failed: {exc}')

    def _collect(self, tmp_dir: str, final: str) -> None:
        pngs = sorted(f for f in os.listdir(tmp_dir) if f.endswith('.png'))
        if not pngs:
            self.get_logger().warning(f'no screenshot appeared in {tmp_dir}')
            return
        os.replace(os.path.join(tmp_dir, pngs[-1]), final)
        self.get_logger().info(f'screenshot -> {final}')


def main() -> None:
    out_dir = sys.argv[1] if len(sys.argv) > 1 else '.'
    os.makedirs(out_dir, exist_ok=True)
    rclpy.init()
    node = StateWatcher(out_dir)
    node.set_parameters([rclpy.parameter.Parameter('use_sim_time', value=True)])
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, rclpy.executors.ExternalShutdownException):
        pass
    finally:
        node.timeline.close()
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
