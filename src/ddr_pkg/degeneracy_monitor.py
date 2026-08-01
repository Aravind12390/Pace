#!/usr/bin/env python3
"""
degeneracy_monitor.py

Standalone scan-matching analytics node. Performs frame-to-frame point-to-plane
alignment of the 3D LiDAR point cloud (projected to the horizontal plane, since
the corridor's structural degeneracy is a planar/axial phenomenon: two long,
flat, parallel walls constrain lateral (Y) translation and yaw well, but leave
translation along the corridor's long axis (X) almost entirely unconstrained).

For a point-to-plane ICP formulation, the Gauss-Newton Hessian

    H = sum_i J_i^T J_i        (Fisher Information Matrix approximation)

is exactly the *Information Matrix* of the scan-matching problem (inverse of
the Cramer-Rao lower bound on pose covariance, under Gaussian residual noise).
Each row of J_i is the point-to-plane Jacobian for correspondence i:

    J_i = [ n_x, n_y, (p_x * n_y - p_y * n_x) ]     (SE(2): x, y, yaw)

where n = (n_x, n_y) is the local surface normal at the matched target point,
and p = (p_x, p_y) is the source point in the sensor frame.

Eigen-decomposing H reveals the degenerate directions of the optimization:
a near-zero eigenvalue means the corresponding eigenvector direction in
(x, y, yaw) space is *not* observable from the current scan geometry -- in a
straight, feature-less corridor this shows up as a small eigenvalue whose
eigenvector aligns with the corridor's long (X) axis.

Published topics:
    /scan_matching/information_matrix   (std_msgs/Float64MultiArray)
        Row-major flattened 3x3 information matrix H (x, y, yaw).

    /scan_matching/degeneracy           (diagnostic_msgs/DiagnosticArray)
        Eigenvalues, condition number, most-degenerate-axis direction, and an
        is_degenerate flag (thresholded on the minimum eigenvalue).

Subscribed:
    /scan/points   (sensor_msgs/PointCloud2)   16-channel 3D LiDAR output
"""

import math

import numpy as np
from scipy.spatial import cKDTree

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSPresetProfiles

from sensor_msgs.msg import PointCloud2
from sensor_msgs_py import point_cloud2 as pc2
from std_msgs.msg import Float64MultiArray, MultiArrayDimension
from diagnostic_msgs.msg import DiagnosticArray, DiagnosticStatus, KeyValue


class DegeneracyMonitor(Node):

    def __init__(self):
        super().__init__('degeneracy_monitor')

        self.declare_parameter('z_band_min', -0.05)   # horizontal-slice band, meters,
        self.declare_parameter('z_band_max', 0.05)     # relative to the lidar_link frame
        self.declare_parameter('max_points', 2000)     # subsample cap for speed
        self.declare_parameter('corr_max_dist', 0.5)   # meters, NN correspondence gate
        self.declare_parameter('normal_k', 8)           # neighbors used for local normal PCA
        self.declare_parameter('icp_iterations', 5)
        self.declare_parameter('degeneracy_eigenvalue_threshold', 5.0)
        self.declare_parameter('corridor_axis', 'x')    # informational label only

        self.z_min = float(self.get_parameter('z_band_min').value)
        self.z_max = float(self.get_parameter('z_band_max').value)
        self.max_points = int(self.get_parameter('max_points').value)
        self.corr_max_dist = float(self.get_parameter('corr_max_dist').value)
        self.normal_k = int(self.get_parameter('normal_k').value)
        self.icp_iters = int(self.get_parameter('icp_iterations').value)
        self.degen_thresh = float(self.get_parameter('degeneracy_eigenvalue_threshold').value)

        qos = QoSPresetProfiles.SENSOR_DATA.value
        self._scan_sub = self.create_subscription(PointCloud2, '/scan/points', self._scan_cb, qos)

        self._info_pub = self.create_publisher(Float64MultiArray, '/scan_matching/information_matrix', 10)
        self._diag_pub = self.create_publisher(DiagnosticArray, '/scan_matching/degeneracy', 10)

        self._prev_xy = None  # Nx2 array, previous scan (target cloud)

        self.get_logger().info(
            'degeneracy_monitor up: point-to-plane scan matching -> '
            'Fisher information matrix H = sum J^T J, eigen-decomposed each scan.'
        )

    # ------------------------------------------------------------------
    def _extract_xy(self, msg: PointCloud2) -> np.ndarray:
        pts = pc2.read_points_numpy(msg, field_names=('x', 'y', 'z'), skip_nans=True)
        if pts.size == 0:
            return np.empty((0, 2))
        band = (pts[:, 2] >= self.z_min) & (pts[:, 2] <= self.z_max)
        pts = pts[band]
        if pts.shape[0] > self.max_points:
            idx = np.random.default_rng().choice(pts.shape[0], self.max_points, replace=False)
            pts = pts[idx]
        return pts[:, :2]

    @staticmethod
    def _local_normals(target_xy: np.ndarray, tree: cKDTree, k: int) -> np.ndarray:
        """Estimate a 2D surface normal at every target point via local PCA."""
        n = target_xy.shape[0]
        normals = np.zeros((n, 2))
        k = min(k, n)
        if k < 3:
            return normals
        _, idxs = tree.query(target_xy, k=k)
        for i in range(n):
            neighborhood = target_xy[idxs[i]]
            centered = neighborhood - neighborhood.mean(axis=0)
            cov = centered.T @ centered
            eigvals, eigvecs = np.linalg.eigh(cov)
            # normal = eigenvector of the *smallest* eigenvalue (least variance
            # direction of the local neighborhood -> perpendicular to the local surface)
            normals[i] = eigvecs[:, 0]
        return normals

    def _point_to_plane_icp(self, source_xy: np.ndarray, target_xy: np.ndarray):
        """
        Small-angle point-to-plane ICP in SE(2). Returns the final Gauss-Newton
        information matrix H = sum J^T J (3x3, order [x, y, yaw]) evaluated at
        the converged alignment.
        """
        tree = cKDTree(target_xy)
        normals = self._local_normals(target_xy, tree, self.normal_k)

        x, y, yaw = 0.0, 0.0, 0.0
        H = np.zeros((3, 3))

        for _ in range(self.icp_iters):
            c, s = math.cos(yaw), math.sin(yaw)
            R = np.array([[c, -s], [s, c]])
            transformed = (R @ source_xy.T).T + np.array([x, y])

            dists, idxs = tree.query(transformed, k=1)
            valid = dists < self.corr_max_dist
            if not np.any(valid):
                break

            p = transformed[valid]
            n = normals[idxs[valid]]
            q = target_xy[idxs[valid]]

            r = np.sum((p - q) * n, axis=1)  # point-to-plane residuals

            # Jacobian rows: d r_i / d[x, y, yaw]
            # For small rotations about the alignment estimate, the yaw column
            # uses the *source* point in the current (rotated) frame:
            px_s, py_s = p[:, 0] - x, p[:, 1] - y
            J = np.column_stack([
                n[:, 0],
                n[:, 1],
                px_s * n[:, 1] - py_s * n[:, 0],
            ])

            H = J.T @ J
            JT_r = J.T @ r
            try:
                delta = -np.linalg.solve(H + 1e-9 * np.eye(3), JT_r)
            except np.linalg.LinAlgError:
                break

            x += delta[0]
            y += delta[1]
            yaw += delta[2]

        return H, (x, y, yaw)

    def _scan_cb(self, msg: PointCloud2):
        cur_xy = self._extract_xy(msg)

        if self._prev_xy is None or self._prev_xy.shape[0] < self.normal_k or cur_xy.shape[0] < 10:
            self._prev_xy = cur_xy
            return

        H, pose_delta = self._point_to_plane_icp(cur_xy, self._prev_xy)
        self._prev_xy = cur_xy

        self._publish_information_matrix(msg, H)
        self._publish_degeneracy_diagnostics(msg, H, pose_delta)

    # ------------------------------------------------------------------
    def _publish_information_matrix(self, msg: PointCloud2, H: np.ndarray):
        out = Float64MultiArray()
        out.layout.dim.append(MultiArrayDimension(label='rows', size=3, stride=9))
        out.layout.dim.append(MultiArrayDimension(label='cols', size=3, stride=3))
        out.data = H.flatten(order='C').tolist()
        self._info_pub.publish(out)

    def _publish_degeneracy_diagnostics(self, msg: PointCloud2, H: np.ndarray, pose_delta):
        eigvals, eigvecs = np.linalg.eigh(H)  # ascending order
        min_eig = float(eigvals[0])
        max_eig = float(eigvals[-1])
        cond = float(max_eig / min_eig) if min_eig > 1e-12 else float('inf')

        degenerate_dir = eigvecs[:, 0]  # eigenvector of smallest eigenvalue
        labels = ['x', 'y', 'yaw']
        dominant_axis = labels[int(np.argmax(np.abs(degenerate_dir)))]

        is_degenerate = min_eig < self.degen_thresh

        diag = DiagnosticArray()
        diag.header = msg.header
        status = DiagnosticStatus()
        status.name = 'scan_matching_degeneracy'
        status.hardware_id = 'lidar_3d'
        status.level = DiagnosticStatus.WARN if is_degenerate else DiagnosticStatus.OK
        status.message = (
            f'DEGENERATE along {dominant_axis} (lambda_min={min_eig:.3f} < '
            f'{self.degen_thresh:.3f})' if is_degenerate else 'well-conditioned'
        )
        status.values = [
            KeyValue(key='lambda_min', value=f'{min_eig:.6f}'),
            KeyValue(key='lambda_mid', value=f'{eigvals[1]:.6f}'),
            KeyValue(key='lambda_max', value=f'{max_eig:.6f}'),
            KeyValue(key='condition_number', value=f'{cond:.3f}'),
            KeyValue(key='degenerate_direction_x', value=f'{degenerate_dir[0]:.4f}'),
            KeyValue(key='degenerate_direction_y', value=f'{degenerate_dir[1]:.4f}'),
            KeyValue(key='degenerate_direction_yaw', value=f'{degenerate_dir[2]:.4f}'),
            KeyValue(key='dominant_degenerate_axis', value=dominant_axis),
            KeyValue(key='is_degenerate', value=str(is_degenerate)),
            KeyValue(key='icp_dx', value=f'{pose_delta[0]:.4f}'),
            KeyValue(key='icp_dy', value=f'{pose_delta[1]:.4f}'),
            KeyValue(key='icp_dyaw', value=f'{pose_delta[2]:.4f}'),
        ]
        diag.status.append(status)
        self._diag_pub.publish(diag)


def main(args=None):
    rclpy.init(args=args)
    node = DegeneracyMonitor()
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
