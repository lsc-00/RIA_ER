#!/usr/bin/env python3

import time

import rclpy
from rclpy.node import Node
from rclpy.qos import (
    QoSProfile,
    QoSReliabilityPolicy,
    QoSDurabilityPolicy,
    QoSHistoryPolicy,
)
from sensor_msgs.msg import Image


class ImageRateMonitor(Node):
    """
    RealSense RGB 토픽을 구독하여 ROS 2/DDS를 통해 실제 수신되는
    이미지 메시지의 FPS와 지연 상태를 측정하는 노드입니다.

    cv_bridge 변환이나 YOLO 추론을 수행하지 않으므로,
    DDS 이미지 전달 성능만 비교할 수 있습니다.
    """

    def __init__(self) -> None:
        super().__init__('image_rate_monitor')

        self.declare_parameter(
            'image_topic',
            '/camera/camera/color/image_raw'
        )
        self.declare_parameter('report_interval', 1.0)
        self.declare_parameter('stall_threshold', 0.5)

        self.image_topic = (
            self.get_parameter('image_topic')
            .get_parameter_value()
            .string_value
        )
        self.report_interval = (
            self.get_parameter('report_interval')
            .get_parameter_value()
            .double_value
        )
        self.stall_threshold = (
            self.get_parameter('stall_threshold')
            .get_parameter_value()
            .double_value
        )

        self.total_frames = 0
        self.interval_frames = 0

        now = time.perf_counter()
        self.start_time = now
        self.last_report_time = now
        self.last_frame_time = None

        self.min_interval = float('inf')
        self.max_interval = 0.0
        self.interval_sum = 0.0
        self.interval_count = 0
        self.stall_count = 0

        image_qos = QoSProfile(
            reliability=QoSReliabilityPolicy.RELIABLE,
            durability=QoSDurabilityPolicy.VOLATILE,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1,
        )

        self.subscription = self.create_subscription(
            Image,
            self.image_topic,
            self.image_callback,
            image_qos
        )

        self.report_timer = self.create_timer(
            self.report_interval,
            self.report_statistics
        )

        self.get_logger().info(
            f'이미지 FPS 측정 시작: {self.image_topic}'
        )
        self.get_logger().info(
            'cv_bridge 및 YOLO 없이 Image 메시지 수신 횟수만 측정합니다.'
        )

    def image_callback(self, msg: Image) -> None:
        """이미지 수신 횟수와 프레임 간격만 기록합니다."""
        del msg

        current_time = time.perf_counter()

        self.total_frames += 1
        self.interval_frames += 1

        if self.last_frame_time is not None:
            frame_interval = current_time - self.last_frame_time

            self.min_interval = min(
                self.min_interval,
                frame_interval
            )
            self.max_interval = max(
                self.max_interval,
                frame_interval
            )
            self.interval_sum += frame_interval
            self.interval_count += 1

            if frame_interval >= self.stall_threshold:
                self.stall_count += 1

                self.get_logger().warning(
                    f'프레임 정지 감지: {frame_interval:.3f}초'
                )

        self.last_frame_time = current_time

    def report_statistics(self) -> None:
        """설정된 주기마다 수신 FPS와 누적 통계를 출력합니다."""
        current_time = time.perf_counter()

        interval_elapsed = current_time - self.last_report_time
        total_elapsed = current_time - self.start_time

        interval_fps = (
            self.interval_frames / interval_elapsed
            if interval_elapsed > 0.0
            else 0.0
        )

        total_fps = (
            self.total_frames / total_elapsed
            if total_elapsed > 0.0
            else 0.0
        )

        if self.interval_count > 0:
            average_interval = (
                self.interval_sum / self.interval_count
            )
            minimum_interval = self.min_interval
        else:
            average_interval = 0.0
            minimum_interval = 0.0

        if self.last_frame_time is None:
            last_frame_age = float('inf')
        else:
            last_frame_age = current_time - self.last_frame_time

        self.get_logger().info(
            '\n'
            f'구간 수신 FPS : {interval_fps:6.2f}\n'
            f'누적 수신 FPS : {total_fps:6.2f}\n'
            f'누적 프레임   : {self.total_frames}\n'
            f'최소 간격     : {minimum_interval:.3f}초\n'
            f'평균 간격     : {average_interval:.3f}초\n'
            f'최대 간격     : {self.max_interval:.3f}초\n'
            f'정지 감지 횟수: {self.stall_count}\n'
            f'마지막 프레임 : {last_frame_age:.3f}초 전'
        )

        self.interval_frames = 0
        self.last_report_time = current_time


def main(args=None) -> None:
    rclpy.init(args=args)

    node = ImageRateMonitor()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('사용자 요청으로 종료합니다.')
    finally:
        node.destroy_node()

        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()