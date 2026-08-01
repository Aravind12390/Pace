#!/usr/bin/env python3
"""
ghost_obstacle_mapper.py  (sub-mapping implementation)

Builds a lifelong 2D occupancy grid from the 3D LiDAR stream and clears the
"ghost obstacle" artifact left behind whenever the dynamic shifting_wall
panel teleports to a new lateral position -- via time-windowed SUB-MAPPING,
not ray-casting.

Why plain ray-casting struggles here
-------------------------------------
A single persistent grid that accumulates log-odds evidence forever has no
natural way to *forget*. Once a cell has been hit by enough beams to become
confidently "occupied", nothing removes that evidence except waiting for
enough future beams to happen to pass through it and slowly outvote the old
evidence -- which can take many scans, and requires an extra heuristic
(a conflict-threshold hack) to even happen at a reasonable rate.

Sub-mapping approach implemented here
---------------------------------------
1. Scans are buffered (not immediately fused into the persistent map) for a
   fixed time window (`submap_window_sec`, default 5 s -- three windows fit
   inside one 15 s wall-shift interval).
2. At the end of each window, a **fresh local submap is rebuilt from
   scratch**, using only that window's scans:
     - a cell gets an OCCUPIED vote every time a beam's endpoint lands in it
     - a cell gets a FREE vote whenever a beam's *measured range at that
       bearing* is farther than the cell's own distance from the sensor
       (i.e. the beam looked straight through that cell and kept going) --
       computed via a per-scan bearing-binned range profile (a "synthetic
       2D scan"), not by walking every intermediate grid cell along each beam.
     - a cell with neither enough occupied nor enough free votes this window
       stays UNRESOLVED for this cycle.
3. Every RESOLVED cell (occupied or free) in the submap's footprint
   **overwrites** the corresponding cell of the persistent map outright.
   UNRESOLVED cells are left untouched (genuinely unobserved regions keep
   whatever the map already said about them).

Because each window is rebuilt purely from its own scans with no memory of
earlier windows, a wall that has moved away simply does not get any
OCCUPIED votes in the first post-move window that observes that space --
it gets FREE votes instead, and the overwrite erases the stale mark in a
single window cycle. There is no decay curve and no conflict-threshold
tuning parameter; "no longer occupied" is proven directly, once, per cycle.

Published:
    /map   (nav_msgs/OccupancyGrid), once per submap commit

Subscribed:
    /scan/points   (sensor_msgs/PointCloud2)
    TF: odom -> lidar_link  (sensor origin/orientation, driven by the fused
                              EKF estimate + robot_state_publisher's static
                              chain from base_footprint up to lidar_link)
"""

import math

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSPresetProfiles, QoSProfile, QoSDurabilityPolicy

from sensor_msgs.msg import PointCloud2
from sensor_msgs_py import point_cloud2 as pc2
from nav_msgs.msg import OccupancyGrid, MapMetaData
from geometry_msgs.msg import Pose

import tf2_ros
from tf2_ros import TransformException


def _yaw_from_quat(q) -> float:
    siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
    cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    return math.atan2(siny_cosp, cosy_cosp)


# ======================================================================
# Pure submap math (no ROS/rclpy dependency -> directly unit-testable)
# ======================================================================

def build_range_profile(points_xy: np.ndarray, sensor_xy, n_bins: int) -> np.ndarray:
    """
    Collapse a scan's hit points into a "synthetic 2D scan": for each of
    n_bins uniform world-frame bearing bins spanning [-pi, pi), the minimum
    range at which *something* was hit in that direction. Bins with no hits
    are left as +inf (nothing seen that way -> no free-space claim possible).
    """
    profile = np.full(n_bins, np.inf, dtype=float)
    if points_xy.shape[0] == 0:
        return profile
    dx = points_xy[:, 0] - sensor_xy[0]
    dy = points_xy[:, 1] - sensor_xy[1]
    ranges = np.hypot(dx, dy)
    bearings = np.arctan2(dy, dx)
    bins = np.clip(((bearings + math.pi) / (2 * math.pi) * n_bins).astype(int), 0, n_bins - 1)
    np.minimum.at(profile, bins, ranges)
    return profile


def vote_scan(patch_x: np.ndarray, patch_y: np.ndarray, points_xy: np.ndarray,
              sensor_xy, n_bins: int, free_margin: float, max_range: float,
              occupied_count: np.ndarray, free_count: np.ndarray,
              resolution: float, patch_origin_rc, world_to_cell_fn):
    """
    Accumulate one scan's votes into the submap's running occupied/free
    count arrays (both same shape as patch_x / patch_y, i.e. the local
    submap patch). Mutates occupied_count / free_count in place.

    patch_x, patch_y : 2D arrays of world-frame cell-center coordinates
                        covering the local submap patch
    points_xy         : this scan's hit points, world frame, shape (N, 2)
    """
    profile = build_range_profile(points_xy, sensor_xy, n_bins)

    # ---- occupied votes: exactly at each hit point's own cell ----
    r0, c0 = patch_origin_rc
    for i in range(points_xy.shape[0]):
        r, c = world_to_cell_fn(points_xy[i, 0], points_xy[i, 1])
        pr, pc = r - r0, c - c0
        if 0 <= pr < occupied_count.shape[0] and 0 <= pc < occupied_count.shape[1]:
            occupied_count[pr, pc] += 1

    # ---- free votes: every patch cell whose own range is comfortably
    #      short of what that bearing's beam actually measured ----
    dx = patch_x - sensor_xy[0]
    dy = patch_y - sensor_xy[1]
    cell_range = np.hypot(dx, dy)
    bearings = np.arctan2(dy, dx)
    bins = np.clip(((bearings + math.pi) / (2 * math.pi) * n_bins).astype(int), 0, n_bins - 1)
    beam_range_at_cell = profile[bins]

    free_mask = (
        np.isfinite(beam_range_at_cell)
        & (cell_range < (beam_range_at_cell - free_margin))
        & (cell_range <= max_range)
    )
    free_count[free_mask] += 1


def resolve_submap(occupied_count: np.ndarray, free_count: np.ndarray,
                    occupied_vote_threshold: int, free_vote_threshold: int) -> np.ndarray:
    """
    Turn a window's accumulated vote counts into a final per-cell state:
    100 = occupied, 0 = free, -1 = unresolved this cycle (left untouched
    in the persistent map). Occupied wins ties against free (a single
    solid-looking hit is stronger evidence than a handful of pass-through
    beams, since real obstacles are typically thin relative to LiDAR beam
    spacing at range).
    """
    resolved = np.full(occupied_count.shape, -1, dtype=np.int8)
    occ = occupied_count >= occupied_vote_threshold
    free = (~occ) & (free_count >= free_vote_threshold)
    resolved[occ] = 100
    resolved[free] = 0
    return resolved


class GhostObstacleMapper(Node):

    def __init__(self):
        super().__init__('ghost_obstacle_mapper')

        # ---------------- grid geometry ----------------
        self.declare_parameter('resolution', 0.05)
        self.declare_parameter('origin_x', -2.0)
        self.declare_parameter('origin_y', -3.0)
        self.declare_parameter('width_m', 24.0)
        self.declare_parameter('height_m', 6.0)

        # ---------------- sub-mapping ----------------
        self.declare_parameter('submap_window_sec', 5.0)         # <= 15s wall-shift interval / 3
        self.declare_parameter('angular_bins', 720)                # 0.5 deg resolution
        self.declare_parameter('free_margin', 0.10)                 # m, safety margin for the free-space test
        self.declare_parameter('occupied_vote_threshold', 3)         # min hits within a window to call a cell occupied
        self.declare_parameter('free_vote_threshold', 3)              # min clean pass-throughs to call a cell free

        # ---------------- sensing ----------------
        self.declare_parameter('z_band_min', -0.05)
        self.declare_parameter('z_band_max', 0.05)
        self.declare_parameter('max_points', 3000)
        self.declare_parameter('max_range', 25.0)
        self.declare_parameter('world_frame', 'odom')
        self.declare_parameter('sensor_frame', 'lidar_link')

        self.res = float(self.get_parameter('resolution').value)
        self.ox = float(self.get_parameter('origin_x').value)
        self.oy = float(self.get_parameter('origin_y').value)
        self.w_m = float(self.get_parameter('width_m').value)
        self.h_m = float(self.get_parameter('height_m').value)
        self.cols = int(round(self.w_m / self.res))
        self.rows = int(round(self.h_m / self.res))

        self.window_sec = float(self.get_parameter('submap_window_sec').value)
        self.n_bins = int(self.get_parameter('angular_bins').value)
        self.free_margin = float(self.get_parameter('free_margin').value)
        self.occ_vote_thresh = int(self.get_parameter('occupied_vote_threshold').value)
        self.free_vote_thresh = int(self.get_parameter('free_vote_threshold').value)

        self.z_min = float(self.get_parameter('z_band_min').value)
        self.z_max = float(self.get_parameter('z_band_max').value)
        self.max_points = int(self.get_parameter('max_points').value)
        self.max_range = float(self.get_parameter('max_range').value)
        self.world_frame = str(self.get_parameter('world_frame').value)
        self.sensor_frame = str(self.get_parameter('sensor_frame').value)

        # persistent map: -1 unknown, 0 free, 100 occupied. Only ever
        # touched by wholesale submap-commit overwrites, never incrementally.
        self.persistent_grid = np.full((self.rows, self.cols), -1, dtype=np.int8)

        self._window_scans = []  # list of (sensor_xy, points_xy) buffered this window

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        qos = QoSPresetProfiles.SENSOR_DATA.value
        self._scan_sub = self.create_subscription(PointCloud2, '/scan/points', self._scan_cb, qos)

        map_qos = QoSProfile(depth=1)
        map_qos.durability = QoSDurabilityPolicy.TRANSIENT_LOCAL
        self._map_pub = self.create_publisher(OccupancyGrid, '/map', map_qos)

        self._commit_timer = self.create_timer(self.window_sec, self._commit_submap)

        self.get_logger().info(
            f"ghost_obstacle_mapper (sub-mapping) up: {self.rows}x{self.cols} grid @ {self.res} m/cell, "
            f"window={self.window_sec}s, occ_vote>={self.occ_vote_thresh}, free_vote>={self.free_vote_thresh}. "
            f"Each window is rebuilt from scratch and wholesale-overwrites the persistent map -- "
            f"no per-cell decay, no conflict-threshold heuristic."
        )

    # ------------------------------------------------------------------
    def _world_to_cell(self, x: float, y: float):
        c = int((x - self.ox) / self.res)
        r = int((y - self.oy) / self.res)
        return r, c

    # ------------------------------------------------------------------
    def _scan_cb(self, msg: PointCloud2):
        try:
            tf = self.tf_buffer.lookup_transform(
                self.world_frame, self.sensor_frame, msg.header.stamp,
                timeout=rclpy.duration.Duration(seconds=0.05),
            )
        except TransformException as ex:
            self.get_logger().debug(f'TF lookup failed, skipping scan: {ex}')
            return

        sx = tf.transform.translation.x
        sy = tf.transform.translation.y
        syaw = _yaw_from_quat(tf.transform.rotation)

        pts = pc2.read_points_numpy(msg, field_names=('x', 'y', 'z'), skip_nans=True)
        if pts.size == 0:
            return
        band = (pts[:, 2] >= self.z_min) & (pts[:, 2] <= self.z_max)
        pts = pts[band]
        if pts.shape[0] == 0:
            return
        if pts.shape[0] > self.max_points:
            idx = np.random.default_rng().choice(pts.shape[0], self.max_points, replace=False)
            pts = pts[idx]

        c_yaw, s_yaw = math.cos(syaw), math.sin(syaw)
        world_x = sx + pts[:, 0] * c_yaw - pts[:, 1] * s_yaw
        world_y = sy + pts[:, 0] * s_yaw + pts[:, 1] * c_yaw
        world_pts = np.column_stack([world_x, world_y])

        # only buffer this scan for the *next* commit; nothing touches the
        # persistent map until the window ends
        self._window_scans.append(((sx, sy), world_pts))

    # ------------------------------------------------------------------
    def _commit_submap(self):
        if not self._window_scans:
            return  # nothing observed this window; leave persistent map untouched

        # ---- local patch bounds: union of this window's sensor footprints ----
        xs = [s[0] for s, _ in self._window_scans]
        ys = [s[1] for s, _ in self._window_scans]
        x_min = max(self.ox, min(xs) - self.max_range)
        x_max = min(self.ox + self.w_m, max(xs) + self.max_range)
        y_min = max(self.oy, min(ys) - self.max_range)
        y_max = min(self.oy + self.h_m, max(ys) + self.max_range)

        r0, c0 = self._world_to_cell(x_min, y_min)
        r1, c1 = self._world_to_cell(x_max, y_max)
        r0, r1 = max(0, r0), min(self.rows, r1 + 1)
        c0, c1 = max(0, c0), min(self.cols, c1 + 1)
        if r1 <= r0 or c1 <= c0:
            self._window_scans = []
            return

        patch_rows, patch_cols = r1 - r0, c1 - c0
        row_idx = np.arange(r0, r1)
        col_idx = np.arange(c0, c1)
        patch_y, patch_x = np.meshgrid(
            self.oy + (row_idx + 0.5) * self.res,
            self.ox + (col_idx + 0.5) * self.res,
            indexing='ij',
        )

        occupied_count = np.zeros((patch_rows, patch_cols), dtype=int)
        free_count = np.zeros((patch_rows, patch_cols), dtype=int)

        for sensor_xy, points_xy in self._window_scans:
            vote_scan(
                patch_x, patch_y, points_xy, sensor_xy, self.n_bins, self.free_margin,
                self.max_range, occupied_count, free_count, self.res, (r0, c0), self._world_to_cell,
            )

        resolved = resolve_submap(occupied_count, free_count, self.occ_vote_thresh, self.free_vote_thresh)

        # ---- wholesale overwrite of the persistent map's local patch ----
        persistent_patch = self.persistent_grid[r0:r1, c0:c1]
        ghost_cleared = int(np.sum((persistent_patch == 100) & (resolved == 0)))
        newly_occupied = int(np.sum((persistent_patch != 100) & (resolved == 100)))
        known = resolved != -1
        persistent_patch[known] = resolved[known]

        n_scans = len(self._window_scans)
        self._window_scans = []

        if ghost_cleared > 0:
            self.get_logger().info(
                f"ghost_obstacle_mapper: submap commit ({n_scans} scans) overwrote "
                f"{ghost_cleared} previously-occupied cells to FREE (ghost clearing) "
                f"and {newly_occupied} cells to newly OCCUPIED"
            )

        self._publish_map()

    # ------------------------------------------------------------------
    def _publish_map(self):
        grid = OccupancyGrid()
        grid.header.stamp = self.get_clock().now().to_msg()
        grid.header.frame_id = self.world_frame
        grid.info = MapMetaData()
        grid.info.resolution = self.res
        grid.info.width = self.cols
        grid.info.height = self.rows
        origin = Pose()
        origin.position.x = self.ox
        origin.position.y = self.oy
        grid.info.origin = origin
        grid.data = self.persistent_grid.flatten(order='C').tolist()
        self._map_pub.publish(grid)


def main(args=None):
    rclpy.init(args=args)
    node = GhostObstacleMapper()
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
