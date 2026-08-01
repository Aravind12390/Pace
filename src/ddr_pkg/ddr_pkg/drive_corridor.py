#!/usr/bin/env python3
"""
drive_corridor.py

Closed-loop driver that steers the DDR robot down the degenerate corridor in a
way that produces BOTH deliverables in a single run:

  1. Three visually-separable trajectories in the X-Y summary plot.

     The slip model in noise_injector.py corrupts *forward velocity only*
     (v_recorded = s(t) * v_true + w_v) and passes angular velocity straight
     through. On a perfectly straight run the degraded dead-reckoning therefore
     keeps yaw == 0 forever and the entire error collapses onto the X axis --
     all three paths trace the identical segment along y=0 and the X-Y plot
     shows one line no matter how large the RMSE is.

     Driving a serpentine fixes this structurally: with reduced v but identical
     omega, the degraded track turns *more sharply per metre travelled*, so it
     peels away from ground truth laterally and the three curves separate.

  2. A populated degeneracy_warnings.log.

     diagnostics_node only writes a warning when degeneracy_monitor is actually
     publishing on /scan_matching/degeneracy AND the dominant degenerate axis
     matches corridor_axis ('x'). This script keeps the robot inside the long
     featureless corridor for the whole run (which is what makes X degenerate),
     and includes a slow "dwell" segment where lambda_min is at its lowest.
     It also *verifies* the pipeline is alive and tells you at the end if it
     is not, instead of leaving you with a silently empty log.

Hazard handled: the corridor.world 'shifting_wall' panel sits at x=10 and
wall_shifter.py toggles it between y=+0.75 and y=-0.75 every 15 s. The panel is
1.5 m wide, so it always blocks one half of the 3 m corridor and the blocked
half alternates -- crossing x=10 open-loop is a coin-flip collision. This script
re-parks the panel to a known side every few seconds (same gz set_pose service
wall_shifter uses, just faster), then routes the robot through the free lane.

Usage (no rebuild needed -- plain rclpy script):

    # terminal 1
    ros2 launch ddr_pkg fusion.launch.py

    # terminal 2, once Gazebo is up and /odom is publishing
    python3 drive_corridor.py

Then generate the plot:

    ros2 run ddr_pkg plot_trajectories \
        --log-dir ~/deliverables --out ~/deliverables/trajectory_comparison.png
"""

import argparse
import math
import subprocess
import sys
import time

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSPresetProfiles

from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist
from diagnostic_msgs.msg import DiagnosticArray


def _yaw_from_quat(q) -> float:
    siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
    cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    return math.atan2(siny_cosp, cosy_cosp)


def _wrap(a: float) -> float:
    return math.atan2(math.sin(a), math.cos(a))


class CorridorDriver(Node):

    # ---- route plan, all distances in metres along the corridor (X) --------
    # phase, x_end, description
    #   weave_a : serpentine before the panel -> builds yaw history, separates
    #             the trajectories well before the slip patch even starts
    #   merge   : converge onto the free lane ahead of the panel
    #   gap     : straight, slow, through the panel gap at x=10
    #   weave_b : serpentine THROUGH the slip patch [10,15] -> this is the
    #             segment that makes the unfused/fused divergence obvious
    #   dwell   : slow crawl in featureless corridor -> lowest lambda_min
    #   run_out : clear the slip patch so the accumulated error is visible

    def __init__(self, args):
        super().__init__('corridor_driver')

        # everything else in the stack runs on sim time; match it
        self.set_parameters([rclpy.parameter.Parameter(
            'use_sim_time', rclpy.Parameter.Type.BOOL, True)])

        self.v_cruise = args.speed
        self.v_slow = args.speed * 0.5
        self.lane_y = args.lane_y
        self.amp = args.amplitude
        self.wavelength = args.wavelength
        self.x_goal = args.x_goal
        self.k_lat = args.k_lat
        self.k_yaw = args.k_yaw
        self.omega_max = args.omega_max

        self.world_name = args.world
        self.panel_model = args.panel_model
        self.panel_x = args.panel_x
        self.panel_park_y = args.panel_park_y
        self.panel_z = args.panel_z
        self.park_wall = not args.no_park_wall

        self._pose = None          # (x, y, yaw)
        self._start_wall_t = time.time()
        self._done = False
        self._degen_msgs = 0
        self._degen_warns = 0
        self._max_x = -1e9

        self._cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.create_subscription(Odometry, '/odom', self._odom_cb, 10)

        # watch the diagnostics pipeline so we can report if it is dead
        self.create_subscription(
            DiagnosticArray, '/scan_matching/degeneracy', self._degen_cb,
            QoSPresetProfiles.SENSOR_DATA.value)
        self.create_subscription(
            DiagnosticArray, '/diagnostics/localization_degeneracy',
            self._warn_cb, 10)

        if self.park_wall:
            self._park_panel()
            # wall_shifter re-toggles every 15 s; override it faster than that
            self.create_timer(3.0, self._park_panel)

        self.create_timer(0.05, self._control_step)   # 20 Hz

        self.get_logger().info(
            f'corridor_driver up: serpentine to x={self.x_goal:.1f} m, '
            f'v={self.v_cruise:.2f} m/s, amplitude={self.amp:.2f} m, '
            f'gap lane y={self.lane_y:+.2f} m'
        )

    # ------------------------------------------------------------------
    def _park_panel(self):
        """Pin the shifting panel to one side so the gap lane stays free."""
        req = (
            f'name: "{self.panel_model}", '
            f'position: {{x: {self.panel_x}, y: {self.panel_park_y}, z: {self.panel_z}}}, '
            f'orientation: {{x: 0, y: 0, z: 0, w: 1}}'
        )
        cmd = [
            'gz', 'service',
            '-s', f'/world/{self.world_name}/set_pose',
            '--reqtype', 'gz.msgs.Pose',
            '--reptype', 'gz.msgs.Boolean',
            '--timeout', '2000',
            '--req', req,
        ]
        try:
            subprocess.run(cmd, capture_output=True, text=True, timeout=4.0)
        except FileNotFoundError:
            self.get_logger().error(
                "'gz' CLI not on PATH - cannot park the shifting panel. "
                "Re-run with --no-park-wall and stop wall_shifter yourself, "
                "or the robot may collide at x=10.")
            self.park_wall = False
        except subprocess.TimeoutExpired:
            self.get_logger().warn('set_pose timed out while parking panel')

    # ------------------------------------------------------------------
    def _degen_cb(self, msg: DiagnosticArray):
        self._degen_msgs += 1

    def _warn_cb(self, msg: DiagnosticArray):
        for st in msg.status:
            if st.level != 0:      # DiagnosticStatus.OK == 0
                self._degen_warns += 1

    def _odom_cb(self, msg: Odometry):
        self._pose = (
            msg.pose.pose.position.x,
            msg.pose.pose.position.y,
            _yaw_from_quat(msg.pose.pose.orientation),
        )

    # ------------------------------------------------------------------
    def _reference(self, x: float):
        """Return (y_ref, v_ref) for the current X along the route."""
        gap_lo, gap_hi = self.panel_x - 1.2, self.panel_x + 0.8

        if x < gap_lo - 1.0:
            # weave_a: serpentine around corridor centre
            y = self.amp * math.sin(2.0 * math.pi * x / self.wavelength)
            return y, self.v_cruise

        if x < gap_lo:
            # merge: blend from the weave onto the free lane
            t = (x - (gap_lo - 1.0)) / 1.0
            y_weave = self.amp * math.sin(2.0 * math.pi * x / self.wavelength)
            return (1.0 - t) * y_weave + t * self.lane_y, self.v_slow

        if x < gap_hi:
            # gap: hold the lane, slow, straight through the panel
            return self.lane_y, self.v_slow

        if x < 15.5:
            # weave_b: serpentine through the slip patch, biased to the
            # free lane side so we never swing back into the panel
            centre = self.lane_y * 0.35
            y = centre + self.amp * math.sin(
                2.0 * math.pi * (x - gap_hi) / self.wavelength)
            v = self.v_slow if 12.0 <= x <= 13.5 else self.v_cruise   # dwell
            return y, v

        # run_out: straighten and clear the patch
        return self.lane_y * 0.35, self.v_cruise

    # ------------------------------------------------------------------
    def _control_step(self):
        if self._done:
            return
        if self._pose is None:
            return

        x, y, yaw = self._pose
        self._max_x = max(self._max_x, x)

        if x >= self.x_goal:
            self._finish('reached goal')
            return

        if time.time() - self._start_wall_t > 900.0:
            self._finish('wall-clock timeout (15 min)')
            return

        y_ref, v_ref = self._reference(x)

        # proportional cross-track -> desired heading -> yaw rate
        lat_err = y_ref - y
        yaw_ref = math.atan(self.k_lat * lat_err)
        yaw_ref = max(-0.6, min(0.6, yaw_ref))
        omega = self.k_yaw * _wrap(yaw_ref - yaw)
        omega = max(-self.omega_max, min(self.omega_max, omega))

        # ease off forward speed while turning hard, keeps it inside the walls
        v = v_ref * max(0.35, 1.0 - 1.1 * abs(omega))

        # hard safety: corridor walls at +/-1.5
        if abs(y) > 1.15:
            v = min(v, self.v_slow * 0.6)

        tw = Twist()
        tw.linear.x = float(v)
        tw.angular.z = float(omega)
        self._cmd_pub.publish(tw)

    # ------------------------------------------------------------------
    def _finish(self, reason: str):
        self._done = True
        stop = Twist()
        for _ in range(10):
            self._cmd_pub.publish(stop)
            time.sleep(0.02)

        self.get_logger().info(f'run complete ({reason}); max x = {self._max_x:.2f} m')

        print('\n' + '=' * 68)
        print(f'  run finished: {reason}')
        print(f'  furthest X reached                : {self._max_x:.2f} m')
        print(f'  /scan_matching/degeneracy msgs    : {self._degen_msgs}')
        print(f'  non-OK localization diagnostics   : {self._degen_warns}')
        print('=' * 68)

        if self._max_x < 15.5:
            print('\n  ! Did not clear the slip patch [10, 15] m.')
            print('    The accumulated slip error will be small and the three')
            print('    trajectories may still look close together.')

        if self._degen_msgs == 0:
            print('\n  ! degeneracy_monitor published NOTHING.')
            print('    diagnostics_node never evaluates its warning branch, so')
            print('    degeneracy_warnings.log will stay empty. Check:')
            print('      ros2 topic hz /scan/points')
            print('      ros2 topic hz /scan_matching/degeneracy')
            print('    An idle /scan/points points at the LiDAR bridge, not')
            print('    at the diagnostics logic.')
        elif self._degen_warns == 0:
            print('\n  ! Scan matching ran but never flagged degeneracy.')
            print('    lambda_min stayed above lambda_min_floor (5.0), or the')
            print('    dominant degenerate axis was not "x". Inspect:')
            print('      ros2 topic echo /scan_matching/degeneracy --once')
            print('    then lower degeneracy_eigenvalue_threshold /')
            print('    lambda_min_floor in fusion.launch.py to match reality.')
        else:
            print('\n  Degeneracy warnings fired. Check:')
            print('      cat ~/deliverables/degeneracy_warnings.log')

        print('\n  Now render the summary plot:')
        print('      ros2 run ddr_pkg plot_trajectories \\')
        print('          --log-dir ~/deliverables \\')
        print('          --out ~/deliverables/trajectory_comparison.png\n')

        raise SystemExit(0)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--speed', type=float, default=0.35, help='cruise m/s')
    p.add_argument('--amplitude', type=float, default=0.55,
                   help='serpentine half-width [m]; corridor is +/-1.5')
    p.add_argument('--wavelength', type=float, default=4.5,
                   help='serpentine period along X [m]')
    p.add_argument('--lane-y', type=float, default=-0.90,
                   help='lateral lane used to pass the panel at x=10')
    p.add_argument('--x-goal', type=float, default=17.5,
                   help='stop once ground-truth X passes this')
    p.add_argument('--k-lat', type=float, default=1.1, help='cross-track gain')
    p.add_argument('--k-yaw', type=float, default=1.8, help='heading gain')
    p.add_argument('--omega-max', type=float, default=0.7, help='rad/s cap')

    p.add_argument('--world', default='corridor_world')
    p.add_argument('--panel-model', default='shifting_wall')
    p.add_argument('--panel-x', type=float, default=10.0)
    p.add_argument('--panel-park-y', type=float, default=0.75,
                   help='park the panel here; +0.75 blocks y in [0, 1.5], '
                        'leaving the negative-y lane free')
    p.add_argument('--panel-z', type=float, default=1.5)
    p.add_argument('--no-park-wall', action='store_true',
                   help='do not touch the shifting panel (stop wall_shifter '
                        'yourself first, or expect a collision at x=10)')

    args = p.parse_args()

    if args.panel_park_y > 0 and args.lane_y > -0.3:
        print('WARNING: panel parked on +y but lane_y is not clearly negative; '
              'the robot may clip the panel at x=10.', file=sys.stderr)

    rclpy.init()
    node = CorridorDriver(args)
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, SystemExit):
        pass
    finally:
        try:
            node._cmd_pub.publish(Twist())
        except Exception:
            pass
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()