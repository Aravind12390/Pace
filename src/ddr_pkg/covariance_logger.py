#!/usr/bin/env python3
"""
covariance_logger.py

Subscribes to the fused EKF output (/odometry/filtered, published by
robot_localization's ekf_node) and:

  * republishes the running planar (x, y, yaw) pose covariance sub-block as
    its own std_msgs/Float64MultiArray topic, for easy plotting/inspection
    without needing to pick apart the full 6x6 nav_msgs/Odometry covariance,
  * logs trace/determinant summary statistics (simple scalar proxies for
    "how uncertain is the filter right now") to the console (throttled),
  * optionally appends a CSV row per message for offline analysis of how
    covariance grows/shrinks as the robot passes through the degenerate
    corridor and the dusty-floor slip patch.

Published:
    /ekf/pose_covariance_planar   (std_msgs/Float64MultiArray)
        Row-major flattened 3x3 covariance over (x, y, yaw), extracted from
        the full 6x6 nav_msgs/Odometry.pose.covariance.

Subscribed:
    /odometry/filtered   (nav_msgs/Odometry)
"""

import csv
import os
import time

import numpy as np
import rclpy
from rclpy.node import Node

from nav_msgs.msg import Odometry
from std_msgs.msg import Float64MultiArray, MultiArrayDimension


# Indices of (x, y, yaw) within ROS's row-major 6x6 [x,y,z,roll,pitch,yaw] covariance
_PLANAR_IDX = [0, 1, 5]


class CovarianceLogger(Node):

    def __init__(self):
        super().__init__('covariance_logger')

        self.declare_parameter('log_to_csv', True)
        self.declare_parameter('csv_path', os.path.expanduser('~/deliverables/ekf_pose_covariance_log.csv'))
        self.declare_parameter('log_throttle_sec', 2.0)

        self.log_to_csv = bool(self.get_parameter('log_to_csv').value)
        self.csv_path = str(self.get_parameter('csv_path').value)
        self.log_throttle_sec = float(self.get_parameter('log_throttle_sec').value)

        self._pub = self.create_publisher(Float64MultiArray, '/ekf/pose_covariance_planar', 10)
        self._sub = self.create_subscription(Odometry, '/odometry/filtered', self._cb, 10)

        self._last_log_time = 0.0
        self._csv_file = None
        self._csv_writer = None
        if self.log_to_csv:
            self._init_csv()

        self.get_logger().info(
            "covariance_logger up: /odometry/filtered -> /ekf/pose_covariance_planar"
            + (f", CSV log at {self.csv_path}" if self.log_to_csv else "")
        )

    def _init_csv(self):
        os.makedirs(os.path.dirname(self.csv_path) or '.', exist_ok=True)
        new_file = not os.path.exists(self.csv_path)
        self._csv_file = open(self.csv_path, 'a', newline='')
        self._csv_writer = csv.writer(self._csv_file)
        if new_file:
            self._csv_writer.writerow([
                'stamp_sec', 'x', 'y', 'yaw',
                'var_x', 'var_y', 'var_yaw',
                'trace', 'determinant',
            ])

    def _cb(self, msg: Odometry):
        full_cov = np.array(msg.pose.covariance, dtype=float).reshape(6, 6)
        planar_cov = full_cov[np.ix_(_PLANAR_IDX, _PLANAR_IDX)]

        out = Float64MultiArray()
        out.layout.dim.append(MultiArrayDimension(label='rows', size=3, stride=9))
        out.layout.dim.append(MultiArrayDimension(label='cols', size=3, stride=3))
        out.data = planar_cov.flatten(order='C').tolist()
        self._pub.publish(out)

        trace = float(np.trace(planar_cov))
        det = float(np.linalg.det(planar_cov))

        now = time.time()
        if now - self._last_log_time >= self.log_throttle_sec:
            self._last_log_time = now
            self.get_logger().info(
                f"pose cov: var_x={planar_cov[0,0]:.5f} var_y={planar_cov[1,1]:.5f} "
                f"var_yaw={planar_cov[2,2]:.5f} trace={trace:.5f} det={det:.3e}"
            )

        if self.log_to_csv and self._csv_writer is not None:
            stamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
            self._csv_writer.writerow([
                f'{stamp:.3f}',
                f'{msg.pose.pose.position.x:.4f}',
                f'{msg.pose.pose.position.y:.4f}',
                '',  # yaw left blank here; extract from quaternion if needed downstream
                f'{planar_cov[0,0]:.6f}', f'{planar_cov[1,1]:.6f}', f'{planar_cov[2,2]:.6f}',
                f'{trace:.6f}', f'{det:.6e}',
            ])
            self._csv_file.flush()

    def destroy_node(self):
        if self._csv_file is not None:
            self._csv_file.close()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = CovarianceLogger()
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
