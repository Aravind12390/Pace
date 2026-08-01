#!/usr/bin/env python3
"""
noise_injector.py

Standalone node that intercepts *ground-truth* sensor streams coming out of
the simulator (the ideal /odom from the gz-sim DiffDrive plugin, and the
/imu stream from the gz-sim IMU sensor) and injects two explicit,
mathematically-specified non-idealities on top of them:

  1. Linear Odometry Slip Model (dynamic, intermittent, patch-based scale slip)

       v_recorded(t) = s(t) * v_true(t) + w_v(t)
       w_v(t) ~ N(0, sigma_v^2)
       s(t)   = slip_scale   if patch_x_min <= Position_X <= patch_x_max
              = 1.0          otherwise

     This is deliberately a pure *forward-velocity* corruption: the wheel
     joints in the physics engine keep spinning at full commanded speed
     (i.e. the ground-truth /odom from gz-sim is untouched), but the
     *reported* linear velocity is scaled down + noised as if the wheels
     were slipping on a dusty patch of floor between x=10 m and x=15 m
     (the "5 m dusty floor patch" called out in the spec). Angular velocity
     is passed straight through, since the spec's model only covers v(t).

  2. IMU Gyro Deterioration Model (Angle Random Walk + Bias Instability / Rate Random Walk)

       omega_measured(t) = omega_true(t) + b(t) + eta_g(t)
       b_dot(t)          = eta_b(t)
       eta_g(t) ~ N(0, sigma_g^2)   (white noise -> Angle Random Walk)
       eta_b(t) ~ N(0, sigma_b^2)   (drives the bias -> Rate Random Walk / bias instability)

     b(t) is integrated online (Euler-Maruyama) as a genuine random walk:
         b(t+dt) = b(t) + N(0, sigma_b^2 * dt)
     applied independently to all three gyro axes (x, y, z), since the
     robot carries a full 6-axis IMU, not just a single yaw gyro.

Published topics:
    /odom_noisy       (nav_msgs/Odometry)      degraded odometry (dead-reckoned
                                                 from the slipped velocity)
    /imu_noisy         (sensor_msgs/Imu)        degraded IMU (drifting gyro)
    /noise/gyro_bias    (geometry_msgs/Vector3Stamped) running bias b(t), for
                                                 inspection/plotting of the
                                                 injected sensor deterioration

Subscribed topics:
    /odom  (nav_msgs/Odometry)   ground-truth odometry from gz-sim DiffDrive
    /imu   (sensor_msgs/Imu)     ground-truth-ish IMU from the gz-sim IMU sensor
"""

import math

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSPresetProfiles

from nav_msgs.msg import Odometry
from sensor_msgs.msg import Imu
from geometry_msgs.msg import Vector3Stamped


def yaw_from_quat(q) -> float:
    """Extract yaw (Z) from a geometry_msgs/Quaternion."""
    siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
    cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    return math.atan2(siny_cosp, cosy_cosp)


def quat_from_yaw(yaw: float):
    return (0.0, 0.0, math.sin(yaw / 2.0), math.cos(yaw / 2.0))


class NoiseInjector(Node):

    def __init__(self):
        super().__init__('noise_injector')

        # ---------------- Parameters: Linear Odometry Slip Model ----------------
        self.declare_parameter('patch_x_min', 10.0)
        self.declare_parameter('patch_x_max', 15.0)
        self.declare_parameter('slip_scale', 0.70)      # s(t) inside the patch
        self.declare_parameter('sigma_v', 0.02)          # w_v std dev [m/s]

        # ---------------- Parameters: IMU deterioration model ----------------
        self.declare_parameter('sigma_g', 0.0025)        # gyro white noise (ARW) std dev [rad/s]
        self.declare_parameter('sigma_b', 1.0e-4)        # bias random-walk driving noise std dev [rad/s / sqrt(s)]
        self.declare_parameter('bias_init', [0.0, 0.0, 0.0])

        self.declare_parameter('random_seed', -1)        # -1 => nondeterministic

        self.patch_x_min = float(self.get_parameter('patch_x_min').value)
        self.patch_x_max = float(self.get_parameter('patch_x_max').value)
        self.slip_scale = float(self.get_parameter('slip_scale').value)
        self.sigma_v = float(self.get_parameter('sigma_v').value)
        self.sigma_g = float(self.get_parameter('sigma_g').value)
        self.sigma_b = float(self.get_parameter('sigma_b').value)

        seed = int(self.get_parameter('random_seed').value)
        self._rng = np.random.default_rng(None if seed < 0 else seed)

        bias0 = self.get_parameter('bias_init').value
        self._gyro_bias = np.array(bias0, dtype=float)  # b(t), 3-vector (x,y,z)

        # Dead-reckoned degraded pose (integrated from v_recorded), independent
        # of the ground-truth pose so slip actually accumulates position error.
        self._noisy_x = None
        self._noisy_y = None
        self._noisy_yaw = None
        self._last_odom_stamp = None

        self._last_imu_stamp = None

        # Track cumulative dead-reckoning position variance for a (very)
        # approximate but monotonically-growing pose covariance estimate,
        # driven by how much slip-scaling / noise has been seen so far.
        self._pos_var = np.array([1e-4, 1e-4])  # var_x, var_y seed values
        self._yaw_var = 1e-5

        qos = QoSPresetProfiles.SENSOR_DATA.value

        self._odom_sub = self.create_subscription(Odometry, '/odom', self._odom_cb, 10)
        self._imu_sub = self.create_subscription(Imu, '/imu', self._imu_cb, qos)

        self._odom_pub = self.create_publisher(Odometry, '/odom_noisy', 10)
        self._imu_pub = self.create_publisher(Imu, '/imu_noisy', qos)
        self._bias_pub = self.create_publisher(Vector3Stamped, '/noise/gyro_bias', 10)

        self.get_logger().info(
            "noise_injector up: slip s(t)=%.2f for x in [%.1f, %.1f] m "
            "(sigma_v=%.3f m/s), gyro ARW sigma_g=%.5f rad/s, "
            "bias RRW sigma_b=%.2e rad/s/sqrt(s)" % (
                self.slip_scale, self.patch_x_min, self.patch_x_max,
                self.sigma_v, self.sigma_g, self.sigma_b
            )
        )

    # ------------------------------------------------------------------
    # Linear Odometry Slip Model
    # ------------------------------------------------------------------
    def _odom_cb(self, msg: Odometry):
        stamp = msg.header.stamp
        t = stamp.sec + stamp.nanosec * 1e-9

        if self._last_odom_stamp is None:
            dt = 0.0
        else:
            dt = max(0.0, t - self._last_odom_stamp)
        self._last_odom_stamp = t

        x_true = msg.pose.pose.position.x
        v_true = msg.twist.twist.linear.x
        omega_true = msg.twist.twist.angular.z  # passed through unmodified (spec covers v(t) only)

        # s(t): step function over the dusty-floor patch
        s_t = self.slip_scale if (self.patch_x_min <= x_true <= self.patch_x_max) else 1.0

        # w_v(t) ~ N(0, sigma_v^2)
        w_v = self._rng.normal(0.0, self.sigma_v)

        v_recorded = s_t * v_true + w_v

        # ---- dead-reckon a *separate* degraded pose from v_recorded ----
        if self._noisy_x is None:
            # seed the degraded track from the first ground-truth pose we see
            self._noisy_x = x_true
            self._noisy_y = msg.pose.pose.position.y
            self._noisy_yaw = yaw_from_quat(msg.pose.pose.orientation)
        else:
            self._noisy_x += v_recorded * math.cos(self._noisy_yaw) * dt
            self._noisy_y += v_recorded * math.sin(self._noisy_yaw) * dt
            self._noisy_yaw += omega_true * dt
            self._noisy_yaw = math.atan2(math.sin(self._noisy_yaw), math.cos(self._noisy_yaw))

            # crude but monotonic uncertainty growth: slip regions and the
            # injected measurement noise both inflate the running covariance
            slip_inflation = 3.0 if s_t < 1.0 else 1.0
            self._pos_var += slip_inflation * (self.sigma_v * max(dt, 1e-3)) ** 2
            self._yaw_var += (1e-4 * max(dt, 1e-3)) ** 2

        out = Odometry()
        out.header = msg.header
        out.header.frame_id = msg.header.frame_id or 'odom'
        out.child_frame_id = msg.child_frame_id or 'base_footprint'

        out.pose.pose.position.x = self._noisy_x
        out.pose.pose.position.y = self._noisy_y
        out.pose.pose.position.z = msg.pose.pose.position.z
        qx, qy, qz, qw = quat_from_yaw(self._noisy_yaw)
        out.pose.pose.orientation.x = qx
        out.pose.pose.orientation.y = qy
        out.pose.pose.orientation.z = qz
        out.pose.pose.orientation.w = qw

        cov = [0.0] * 36
        cov[0] = float(self._pos_var[0])   # x
        cov[7] = float(self._pos_var[1])   # y
        cov[14] = 1e-6                      # z (unused, planar)
        cov[35] = float(self._yaw_var)      # yaw
        out.pose.covariance = cov

        out.twist.twist.linear.x = v_recorded
        out.twist.twist.angular.z = omega_true
        tcov = [0.0] * 36
        tcov[0] = self.sigma_v ** 2
        tcov[35] = 1e-4
        out.twist.covariance = tcov

        self._odom_pub.publish(out)

    # ------------------------------------------------------------------
    # IMU Gyro Deterioration Model: Angle Random Walk + Bias Instability
    # ------------------------------------------------------------------
    def _imu_cb(self, msg: Imu):
        stamp = msg.header.stamp
        t = stamp.sec + stamp.nanosec * 1e-9
        if self._last_imu_stamp is None:
            dt = 0.0
        else:
            dt = max(0.0, t - self._last_imu_stamp)
        self._last_imu_stamp = t

        omega_true = np.array([
            msg.angular_velocity.x,
            msg.angular_velocity.y,
            msg.angular_velocity.z,
        ])

        # ---- bias random walk: b(t+dt) = b(t) + eta_b(t) ; eta_b ~ N(0, sigma_b^2) ----
        # Euler-Maruyama integration of the SDE b_dot = eta_b(t):
        # the driving-noise increment over an interval dt has variance
        # sigma_b^2 * dt (standard discretization of a continuous random walk).
        if dt > 0.0:
            self._gyro_bias += self._rng.normal(0.0, self.sigma_b * math.sqrt(dt), size=3)

        # ---- white measurement noise: eta_g(t) ~ N(0, sigma_g^2) ----
        eta_g = self._rng.normal(0.0, self.sigma_g, size=3)

        omega_measured = omega_true + self._gyro_bias + eta_g

        out = Imu()
        out.header = msg.header

        out.angular_velocity.x = float(omega_measured[0])
        out.angular_velocity.y = float(omega_measured[1])
        out.angular_velocity.z = float(omega_measured[2])
        gcov = self.sigma_g ** 2
        out.angular_velocity_covariance = [
            gcov, 0.0, 0.0,
            0.0, gcov, 0.0,
            0.0, 0.0, gcov,
        ]

        # linear acceleration / orientation are outside this model's scope;
        # pass through whatever the (already sensor-noise-modeled) source gives us
        out.linear_acceleration = msg.linear_acceleration
        out.linear_acceleration_covariance = msg.linear_acceleration_covariance
        out.orientation = msg.orientation
        out.orientation_covariance = msg.orientation_covariance

        self._imu_pub.publish(out)

        bias_msg = Vector3Stamped()
        bias_msg.header = msg.header
        bias_msg.vector.x = float(self._gyro_bias[0])
        bias_msg.vector.y = float(self._gyro_bias[1])
        bias_msg.vector.z = float(self._gyro_bias[2])
        self._bias_pub.publish(bias_msg)


def main(args=None):
    rclpy.init(args=args)
    node = NoiseInjector()
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
