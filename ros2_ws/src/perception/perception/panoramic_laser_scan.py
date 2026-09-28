#!/usr/bin/env python3
import math
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import LaserScan
from message_filters import ApproximateTimeSynchronizer, Subscriber
import tf2_ros
from geometry_msgs.msg import TransformStamped
import numpy as np


class PanoramicLaserScan(Node):
    """Merges front and back laser scans into a single 360° LaserScan topic.

    Both lasers are transformed into a common frame (base_link by default)
    using their TF transforms, then their readings are merged into a unified
    angular grid covering [−π, π].
    """

    def __init__(self):
        super().__init__('panoramic_laser_scan')

        self.declare_parameter('output_frame', 'base_footprint')
        self.declare_parameter('output_topic', '/panoramic/laser_scan')
        self.declare_parameter('scan_topic_1', '/front_laser/laser_scan')
        self.declare_parameter('scan_topic_2', '/back_laser/laser_scan')
        self.declare_parameter('num_rays', 360)
        self.declare_parameter('sync_slop', 0.1)  # seconds

        self._output_frame = self.get_parameter('output_frame').value
        self._angle_increment = 2.0 * math.pi / self.get_parameter('num_rays').value
        sync_slop = self.get_parameter('sync_slop').value

        self._pub = self.create_publisher(
            LaserScan,
            self.get_parameter('output_topic').value,
            10,
        )

        self._tf_buffer = tf2_ros.Buffer()
        self._tf_listener = tf2_ros.TransformListener(self._tf_buffer, self)

        qos = QoSProfile(depth=10, reliability=ReliabilityPolicy.BEST_EFFORT)

        front_sub = Subscriber(self, LaserScan, self.get_parameter('scan_topic_1').value, qos_profile=qos)
        back_sub = Subscriber(self, LaserScan, self.get_parameter('scan_topic_2').value, qos_profile=qos)

        self._sync = ApproximateTimeSynchronizer(
            [front_sub, back_sub],
            queue_size=10,
            slop=sync_slop,
        )
        self._sync.registerCallback(self._merge_callback)

        self.get_logger().info('PanoramicLaserScan node started.')

    @staticmethod
    def _quat_to_rotation_matrix(q) -> np.ndarray:
        """Full 3x3 rotation matrix from a quaternion (handles any orientation)."""
        x, y, z, w = q.x, q.y, q.z, q.w
        return np.array([
            [1 - 2*(y*y + z*z),   2*(x*y - w*z),       2*(x*z + w*y)],
            [2*(x*y + w*z),       1 - 2*(x*x + z*z),   2*(y*z - w*x)],
            [2*(x*z - w*y),       2*(y*z + w*x),       1 - 2*(x*x + y*y)],
        ])

    def _scan_to_cartesian(self, scan: LaserScan, transform: TransformStamped):
        """Convert LaserScan ranges to (x, y) points in the output frame.

        Uses the full 3D rotation matrix so that laser frames with z pointing
        downward (or any non-standard orientation) are handled correctly.
        Only the x/y components after rotation are used — the scan is assumed
        planar in its own frame.
        """
        ranges = np.array(scan.ranges)
        n = len(ranges)

        angles = np.linspace(scan.angle_min, scan.angle_max, n)
        valid = np.isfinite(ranges) & (ranges >= scan.range_min) & (ranges <= scan.range_max)

        # Points in the laser's own frame (z=0 plane of that frame)
        lx = ranges * np.cos(angles)
        ly = ranges * np.sin(angles)
        lz = np.zeros(n)
        points_local = np.stack([lx, ly, lz], axis=0)  # (3, N)

        R = self._quat_to_rotation_matrix(transform.transform.rotation)
        t = transform.transform.translation

        points_world = R @ points_local  # (3, N)
        xs = points_world[0] + t.x
        ys = points_world[1] + t.y

        return xs[valid], ys[valid]

    def _merge_callback(self, front_msg: LaserScan, back_msg: LaserScan):
        try:
            front_tf = self._tf_buffer.lookup_transform(
                self._output_frame,
                front_msg.header.frame_id,
                rclpy.time.Time(),
            )
            back_tf = self._tf_buffer.lookup_transform(
                self._output_frame,
                back_msg.header.frame_id,
                rclpy.time.Time(),
            )
        except (tf2_ros.LookupException, tf2_ros.ExtrapolationException) as e:
            self.get_logger().warn(f'TF lookup failed: {e}', throttle_duration_sec=2.0)
            return

        fx, fy = self._scan_to_cartesian(front_msg, front_tf)
        bx, by = self._scan_to_cartesian(back_msg, back_tf)

        xs = np.concatenate([fx, bx])
        ys = np.concatenate([fy, by])

        # Determine range limits from both scanners
        range_min = min(front_msg.range_min, back_msg.range_min)
        range_max = max(front_msg.range_max, back_msg.range_max)

        n_bins = int(round(2.0 * math.pi / self._angle_increment))
        output_ranges = np.full(n_bins, float('inf'))

        angles_out = np.arctan2(ys, xs)
        dists_out = np.hypot(xs, ys)

        for angle, dist in zip(angles_out, dists_out):
            idx = int(round((angle + math.pi) / self._angle_increment)) % n_bins
            if dist < output_ranges[idx]:
                output_ranges[idx] = dist

        # Leave empty bins as inf — slam_toolbox and nav2 treat inf as "no obstacle",
        # whereas NaN is rejected as an invalid reading.

        out = LaserScan()
        out.header.stamp = front_msg.header.stamp
        out.header.frame_id = self._output_frame
        out.angle_min = -math.pi
        out.angle_max = math.pi - self._angle_increment
        out.angle_increment = self._angle_increment
        out.time_increment = 0.0
        out.scan_time = max(front_msg.scan_time, back_msg.scan_time)
        out.range_min = range_min
        out.range_max = range_max
        out.ranges = output_ranges.tolist()

        self._pub.publish(out)


def main(args=None):
    rclpy.init(args=args)
    node = PanoramicLaserScan()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
