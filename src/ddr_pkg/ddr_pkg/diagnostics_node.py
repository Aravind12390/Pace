#!/usr/bin/env python3
import csv
import os
import time
from collections import deque

import numpy as np
import rclpy
from rclpy.node import Node

from nav_msgs.msg import Odometry
from diagnostic_msgs.msg import DiagnosticArray, DiagnosticStatus, KeyValue

_PLANAR_IDX = [0, 1, 5]  # x, y, yaw within ROS's 6x6 row-major covariance


def _kv_to_dict(values):
    return {kv.key: kv.value for kv in values}


class DiagnosticsNode(Node):

    def __init__(self):
        super().__init__('diagnostics_node')

        self.declare_parameter('corridor_axis', 'x')
        self.declare_parameter('lambda_min_floor', 5.0)         # sustained-degeneracy hard threshold
        self.declare_parameter('spike_window', 10)                # samples in the rolling baseline
        self.declare_parameter('spike_drop_ratio', 0.35)          # trigger if lambda_min < ratio * rolling_mean
        self.declare_parameter('warn_throttle_sec', 3.0)          # avoid log-spamming while sustained
        self.declare_parameter('csv_path', os.path.expanduser('~/deliverables/diagnostics_log.csv'))
        self.declare_parameter('warnings_log_path', os.path.expanduser('~/deliverables/degeneracy_warnings.log'))

        self.corridor_axis = str(self.get_parameter('corridor_axis').value)
        self.lambda_min_floor = float(self.get_parameter('lambda_min_floor').value)
        self.spike_window = int(self.get_parameter('spike_window').value)
        self.spike_drop_ratio = float(self.get_parameter('spike_drop_ratio').value)
        self.warn_throttle_sec = float(self.get_parameter('warn_throttle_sec').value)
        self.csv_path = str(self.get_parameter('csv_path').value)
        self.warn_log_path = str(self.get_parameter('warnings_log_path').value)

        self._lambda_min_history = deque(maxlen=self.spike_window)
        self._latest_scan_diag = None   # dict of most recent scan_matching/degeneracy KeyValues
        self._last_warn_time = 0.0

        self._diag_pub = self.create_publisher(DiagnosticArray, '/diagnostics/localization_degeneracy', 10)

        self._odom_sub = self.create_subscription(Odometry, '/odometry/filtered', self._odom_cb, 10)
        self._scan_diag_sub = self.create_subscription(
            DiagnosticArray, '/scan_matching/degeneracy', self._scan_diag_cb, 10
        )

        self._init_csv()
        self._init_warn_log()

        self.get_logger().info(
            f"diagnostics_node up: corridor_axis='{self.corridor_axis}', "
            f"lambda_min_floor={self.lambda_min_floor}, spike_drop_ratio={self.spike_drop_ratio}, "
            f"csv={self.csv_path}, warnings_log={self.warn_log_path}"
        )

    # ------------------------------------------------------------------
    def _init_csv(self):
        os.makedirs(os.path.dirname(self.csv_path) or '.', exist_ok=True)
        new_file = not os.path.exists(self.csv_path)
        self._csv_file = open(self.csv_path, 'a', newline='')
        self._csv_writer = csv.writer(self._csv_file)
        if new_file:
            self._csv_writer.writerow([
                'timestamp',
                'cov_xx', 'cov_xy', 'cov_xyaw',
                'cov_yx', 'cov_yy', 'cov_yyaw',
                'cov_yawx', 'cov_yawy', 'cov_yawyaw',
                'lambda_min', 'lambda_mid', 'lambda_max', 'condition_number',
                'dominant_degenerate_axis', 'is_degenerate_scanmatch',
                'warning_triggered', 'warning_type', 'warning_message',
            ])

    def _init_warn_log(self):
        os.makedirs(os.path.dirname(self.warn_log_path) or '.', exist_ok=True)
        self._warn_file = open(self.warn_log_path, 'a')

    # ------------------------------------------------------------------
    def _scan_diag_cb(self, msg: DiagnosticArray):
        if not msg.status:
            return
        self._latest_scan_diag = _kv_to_dict(msg.status[0].values)
        lam_min = float(self._latest_scan_diag.get('lambda_min', 'nan'))
        if not math_isnan(lam_min):
            self._lambda_min_history.append(lam_min)

    def _odom_cb(self, msg: Odometry):
        full_cov = np.array(msg.pose.covariance, dtype=float).reshape(6, 6)
        planar_cov = full_cov[np.ix_(_PLANAR_IDX, _PLANAR_IDX)]

        stamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9

        warning_triggered = False
        warning_type = ''
        warning_message = ''

        lam_min = float('nan')
        lam_mid = float('nan')
        lam_max = float('nan')
        cond = float('nan')
        dom_axis = 'unknown'
        is_degenerate_scanmatch = 'unknown'

        if self._latest_scan_diag is not None:
            d = self._latest_scan_diag
            lam_min = float(d.get('lambda_min', 'nan'))
            lam_mid = float(d.get('lambda_mid', 'nan'))
            lam_max = float(d.get('lambda_max', 'nan'))
            cond = float(d.get('condition_number', 'nan'))
            dom_axis = d.get('dominant_degenerate_axis', 'unknown')
            is_degenerate_scanmatch = d.get('is_degenerate', 'unknown')

            axis_matches = (dom_axis == self.corridor_axis)

            sustained = (not math_isnan(lam_min)) and lam_min < self.lambda_min_floor and axis_matches

            spiking = False
            if len(self._lambda_min_history) >= max(3, self.spike_window // 2):
                baseline = float(np.mean(list(self._lambda_min_history)[:-1])) if len(self._lambda_min_history) > 1 else lam_min
                if baseline > 1e-9 and axis_matches and lam_min < self.spike_drop_ratio * baseline:
                    spiking = True

            if sustained or spiking:
                warning_triggered = True
                warning_type = 'SPIKE' if spiking else 'SUSTAINED'
                warning_message = (
                    f"LOCALIZATION_DEGENERACY_WARNING: {warning_type} degeneracy along "
                    f"'{dom_axis}' axis -- lambda_min={lam_min:.4f} "
                    f"(floor={self.lambda_min_floor:.2f}), condition_number={cond:.2f}, "
                    f"fused_var_{self.corridor_axis}={self._axis_variance(planar_cov):.5f}"
                )
                self._emit_warning(warning_message)

        self._log_row(
            stamp, planar_cov, lam_min, lam_mid, lam_max, cond,
            dom_axis, is_degenerate_scanmatch,
            warning_triggered, warning_type, warning_message,
        )
        self._publish_diagnostic(msg.header, warning_triggered, warning_type, warning_message,
                                  lam_min, cond, dom_axis)

    def _axis_variance(self, planar_cov: np.ndarray) -> float:
        idx = {'x': 0, 'y': 1, 'yaw': 2}.get(self.corridor_axis, 0)
        return float(planar_cov[idx, idx])

    def _emit_warning(self, message: str):
        now = time.time()
        if now - self._last_warn_time >= self.warn_throttle_sec:
            self._last_warn_time = now
            self.get_logger().warn(message)
            self._warn_file.write(f'{time.time():.3f} {message}\n')
            self._warn_file.flush()

    def _log_row(self, stamp, planar_cov, lam_min, lam_mid, lam_max, cond,
                 dom_axis, is_degenerate_scanmatch, warning_triggered, warning_type, warning_message):
        self._csv_writer.writerow([
            f'{stamp:.3f}',
            f'{planar_cov[0,0]:.6f}', f'{planar_cov[0,1]:.6f}', f'{planar_cov[0,2]:.6f}',
            f'{planar_cov[1,0]:.6f}', f'{planar_cov[1,1]:.6f}', f'{planar_cov[1,2]:.6f}',
            f'{planar_cov[2,0]:.6f}', f'{planar_cov[2,1]:.6f}', f'{planar_cov[2,2]:.6f}',
            f'{lam_min:.6f}' if not math_isnan(lam_min) else '',
            f'{lam_mid:.6f}' if not math_isnan(lam_mid) else '',
            f'{lam_max:.6f}' if not math_isnan(lam_max) else '',
            f'{cond:.6f}' if not math_isnan(cond) else '',
            dom_axis, is_degenerate_scanmatch,
            warning_triggered, warning_type, warning_message,
        ])
        self._csv_file.flush()

    def _publish_diagnostic(self, header, warning_triggered, warning_type, warning_message,
                             lam_min, cond, dom_axis):
        diag = DiagnosticArray()
        diag.header = header
        status = DiagnosticStatus()
        status.name = 'localization_degeneracy'
        status.hardware_id = 'ekf_filter_node+lidar_3d'
        status.level = DiagnosticStatus.WARN if warning_triggered else DiagnosticStatus.OK
        status.message = warning_message if warning_triggered else 'nominal'
        status.values = [
            KeyValue(key='warning_type', value=warning_type),
            KeyValue(key='lambda_min', value=f'{lam_min:.6f}' if not math_isnan(lam_min) else ''),
            KeyValue(key='condition_number', value=f'{cond:.6f}' if not math_isnan(cond) else ''),
            KeyValue(key='dominant_degenerate_axis', value=dom_axis),
        ]
        diag.status.append(status)
        self._diag_pub.publish(diag)

    def destroy_node(self):
        try:
            self._csv_file.close()
            self._warn_file.close()
        except Exception:
            pass
        super().destroy_node()


def math_isnan(x) -> bool:
    try:
        return x != x  # NaN != NaN
    except TypeError:
        return True


def main(args=None):
    rclpy.init(args=args)
    node = DiagnosticsNode()
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
