#!/usr/bin/env python3
import subprocess

import rclpy
from rclpy.node import Node


class WallShifter(Node):

    def __init__(self):
        super().__init__('wall_shifter')

        self.declare_parameter('world_name', 'corridor_world')
        self.declare_parameter('model_name', 'shifting_wall')
        self.declare_parameter('period_sec', 15.0)
        self.declare_parameter('shift_amount', 1.5)   # meters, sideways (Y)
        self.declare_parameter('x', 10.0)              # halfway down a 20 m hall
        self.declare_parameter('z', 1.5)
        self.declare_parameter('start_y', 0.75)
        self.declare_parameter('service_timeout_ms', 2000)
        # NOTE: 'use_sim_time' is declared automatically by rclpy.Node itself
        # (it's a standard ROS 2 parameter every node gets); declaring it again
        # here raises ParameterAlreadyDeclaredException, so it's intentionally
        # not redeclared. It's still set fine via the launch file's parameters=[...].

        self.world_name = self.get_parameter('world_name').value
        self.model_name = self.get_parameter('model_name').value
        self.period_sec = float(self.get_parameter('period_sec').value)
        self.shift_amount = float(self.get_parameter('shift_amount').value)
        self.x = float(self.get_parameter('x').value)
        self.z = float(self.get_parameter('z').value)
        self.timeout_ms = int(self.get_parameter('service_timeout_ms').value)

        start_y = float(self.get_parameter('start_y').value)
        # Two alternating lateral positions, exactly `shift_amount` apart.
        self._y_states = [start_y, start_y - self.shift_amount]
        self._state_idx = 0

        self.service_name = f'/world/{self.world_name}/set_pose'

        self.get_logger().info(
            f"wall_shifter started: model='{self.model_name}' will shift "
            f"{self.shift_amount:.2f} m sideways every {self.period_sec:.1f} s "
            f"(y in {self._y_states}) via service '{self.service_name}'"
        )

        self._timer = self.create_timer(self.period_sec, self._on_timer)

    def _on_timer(self):
        self._state_idx = (self._state_idx + 1) % len(self._y_states)
        y = self._y_states[self._state_idx]
        self._set_pose(self.model_name, self.x, y, self.z)

    def _set_pose(self, name: str, x: float, y: float, z: float):
        req = (
            f'name: "{name}", '
            f'position: {{x: {x}, y: {y}, z: {z}}}, '
            f'orientation: {{x: 0, y: 0, z: 0, w: 1}}'
        )
        cmd = [
            'gz', 'service',
            '-s', self.service_name,
            '--reqtype', 'gz.msgs.Pose',
            '--reptype', 'gz.msgs.Boolean',
            '--timeout', str(self.timeout_ms),
            '--req', req,
        ]
        try:
            result = subprocess.run(
                cmd, capture_output=True, text=True, timeout=self.timeout_ms / 1000.0 + 2.0
            )
            if result.returncode == 0:
                self.get_logger().info(
                    f"Shifted '{name}' to (x={x:.2f}, y={y:.2f}, z={z:.2f})"
                )
            else:
                self.get_logger().warn(
                    f"set_pose call failed (rc={result.returncode}): "
                    f"{result.stderr.strip() or result.stdout.strip()}"
                )
        except FileNotFoundError:
            self.get_logger().error(
                "'gz' CLI not found. Ensure a Gazebo (gz-sim) install providing "
                "the 'gz service' command is on PATH."
            )
        except subprocess.TimeoutExpired:
            self.get_logger().warn(f"set_pose call to '{self.service_name}' timed out")


def main(args=None):
    rclpy.init(args=args)
    node = WallShifter()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
