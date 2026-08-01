#!/usr/bin/env python3
import csv
import math
import os

import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry


def _yaw_from_quat(q) -> float:
    siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
    cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    return math.atan2(siny_cosp, cosy_cosp)


class TrajectoryLogger(Node):

    def __init__(self):
        super().__init__('trajectory_logger')

        self.declare_parameter('gt_topic', '/odom')
        self.declare_parameter('unfused_topic', '/odom_noisy')
        self.declare_parameter('fused_topic', '/odometry/filtered')
        self.declare_parameter('log_dir', os.path.expanduser('~/deliverables'))

        log_dir = str(self.get_parameter('log_dir').value)
        os.makedirs(log_dir, exist_ok=True)

        self._files = {}
        self._writers = {}
        for key, fname in (
            ('gt', 'gt_trajectory.csv'),
            ('unfused', 'unfused_trajectory.csv'),
            ('fused', 'fused_trajectory.csv'),
        ):
            path = os.path.join(log_dir, fname)
            new_file = not os.path.exists(path)
            f = open(path, 'a', newline='')
            w = csv.writer(f)
            if new_file:
                w.writerow(['timestamp', 'x', 'y', 'yaw'])
            self._files[key] = f
            self._writers[key] = w

        gt_topic = str(self.get_parameter('gt_topic').value)
        unfused_topic = str(self.get_parameter('unfused_topic').value)
        fused_topic = str(self.get_parameter('fused_topic').value)

        self.create_subscription(Odometry, gt_topic, lambda m: self._cb(m, 'gt'), 10)
        self.create_subscription(Odometry, unfused_topic, lambda m: self._cb(m, 'unfused'), 10)
        self.create_subscription(Odometry, fused_topic, lambda m: self._cb(m, 'fused'), 10)

        self.get_logger().info(
            f"trajectory_logger up: {gt_topic} -> gt_trajectory.csv, "
            f"{unfused_topic} -> unfused_trajectory.csv, "
            f"{fused_topic} -> fused_trajectory.csv (dir={log_dir})"
        )

    def _cb(self, msg: Odometry, key: str):
        stamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y
        yaw = _yaw_from_quat(msg.pose.pose.orientation)
        self._writers[key].writerow([f'{stamp:.3f}', f'{x:.4f}', f'{y:.4f}', f'{yaw:.4f}'])
        self._files[key].flush()

    def destroy_node(self):
        for f in self._files.values():
            try:
                f.close()
            except Exception:
                pass
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = TrajectoryLogger()
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
