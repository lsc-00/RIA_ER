#!/usr/bin/env python3
"""
YOLO 2D 검출 결과 (u, v)와 RealSense aligned depth를 결합하여
카메라 optical frame 기준 3차원 좌표 (X, Y, Z)를 계산하는 ROS 2 노드입니다.

입력 Topic
---------
1) /fried_food_vision/detections
   - vision_msgs/msg/Detection2DArray
   - yolo_detection_node가 publish하는 2D 검출 결과
   - bbox.center.position.x/y가 객체 중심 픽셀 (u, v)

2) /camera/camera/aligned_depth_to_color/image_raw
   - sensor_msgs/msg/Image
   - Color 영상 좌표계로 정렬된 Depth 영상

3) /camera/camera/color/camera_info
   - sensor_msgs/msg/CameraInfo
   - Color 카메라 내부 파라미터 fx, fy, cx, cy

출력 Topic
---------
1) /fried_food_vision/camera_positions
   - geometry_msgs/msg/PoseArray
   - Depth를 정상적으로 얻은 객체들의 카메라 기준 XYZ 위치

2) /fried_food_vision/detections_3d
   - vision_msgs/msg/Detection3DArray
   - class / confidence와 카메라 기준 XYZ를 함께 보존한 3D 검출 결과

좌표계
-----
RealSense optical frame / ROS Image convention:
    +X : 영상 오른쪽
    +Y : 영상 아래쪽
    +Z : 카메라 전방

주의
----
이 XYZ는 로봇 Base 좌표가 아닙니다.
Elite CS612 좌표로 사용하려면 이후 Hand-eye Calibration / TF 변환이 필요합니다.
"""

from collections import deque
import copy
import math
import time
from typing import Deque, Optional, Tuple

import cv2
import numpy as np
import rclpy

from cv_bridge import CvBridge, CvBridgeError
from geometry_msgs.msg import Pose, PoseArray
from rclpy.node import Node
from rclpy.qos import (
    QoSDurabilityPolicy,
    QoSHistoryPolicy,
    QoSProfile,
    QoSReliabilityPolicy,
)
from sensor_msgs.msg import CameraInfo, Image
from vision_msgs.msg import (
    Detection2DArray,
    Detection3D,
    Detection3DArray,
)


class DepthPositionNode(Node):
    """
    YOLO의 2D Bounding Box 중심 픽셀 (u, v)을
    aligned depth와 CameraInfo를 이용해 카메라 좌표 (X, Y, Z)로 변환합니다.

    전체 처리 순서
    -------------
    RealSense
      ├─ aligned depth ───────────────┐
      └─ camera_info ─────────────┐   │
                                  │   │
    yolo_detection_node           │   │
      └─ Detection2DArray ────────┼───┤
                                  ▼   ▼
                            DepthPositionNode
                                  │
                                  ├─ 중심 픽셀 (u, v)
                                  ├─ 주변 Depth 중앙값 Z
                                  ├─ CameraInfo K 행렬
                                  │
                                  └─ X, Y, Z 계산
                                          │
                                          ▼
                          /fried_food_vision/camera_positions
    """

    def __init__(self) -> None:
        super().__init__('depth_position_node')

        # ==============================================================
        # [1] ROS 파라미터
        # ==============================================================

        self.declare_parameter(
            'detections_topic',
            '/fried_food_vision/detections',
        )

        self.declare_parameter(
            'aligned_depth_topic',
            '/camera/camera/aligned_depth_to_color/image_raw',
        )

        self.declare_parameter(
            'camera_info_topic',
            '/camera/camera/color/camera_info',
        )

        self.declare_parameter(
            'positions_topic',
            '/fried_food_vision/camera_positions',
        )

        self.declare_parameter(
            'detections_3d_topic',
            '/fried_food_vision/detections_3d',
        )

        # 중심 픽셀 한 점만 사용하면 반사/결측에 민감하므로
        # 중심 주변 ROI의 Depth 중앙값을 사용합니다.
        # radius=4 -> 9 x 9 영역
        self.declare_parameter(
            'depth_radius',
            4,
        )

        self.declare_parameter(
            'min_depth_m',
            0.10,
        )

        self.declare_parameter(
            'max_depth_m',
            3.00,
        )

        # RealSense ROS depth가 16UC1일 때 기본적으로
        # raw depth 값을 meter로 바꾸기 위한 배율입니다.
        # 현재 기본값 0.001은 1 raw unit = 1 mm로 해석합니다.
        # 실제 topic 단위를 확인했는데 다르다면 이 값만 변경하면 됩니다.
        self.declare_parameter(
            'depth_scale',
            0.001,
        )

        # YOLO 검출 결과의 원본 RGB timestamp와
        # aligned depth timestamp의 최대 허용 차이입니다.
        # 프레임이 너무 오래 차이나면 잘못된 Depth를 붙이지 않도록 건너뜁니다.
        self.declare_parameter(
            'sync_tolerance_sec',
            0.10,
        )

        # 최근 Depth 프레임을 몇 장 보관할지 결정합니다.
        # YOLO 추론 지연이 있어도 원본 RGB timestamp와 가장 가까운
        # 과거 Depth를 찾기 위해 작은 버퍼를 유지합니다.
        self.declare_parameter(
            'depth_buffer_size',
            30,
        )

        # 터미널 출력 간격입니다.
        self.declare_parameter(
            'print_interval_sec',
            0.20,
        )

        # ==============================================================
        # [2] 파라미터 읽기
        # ==============================================================

        self.detections_topic = self.get_parameter(
            'detections_topic'
        ).value

        self.aligned_depth_topic = self.get_parameter(
            'aligned_depth_topic'
        ).value

        self.camera_info_topic = self.get_parameter(
            'camera_info_topic'
        ).value

        self.positions_topic = self.get_parameter(
            'positions_topic'
        ).value

        self.detections_3d_topic = self.get_parameter(
            'detections_3d_topic'
        ).value

        self.depth_radius = int(
            self.get_parameter('depth_radius').value
        )

        self.min_depth_m = float(
            self.get_parameter('min_depth_m').value
        )

        self.max_depth_m = float(
            self.get_parameter('max_depth_m').value
        )

        self.depth_scale = float(
            self.get_parameter('depth_scale').value
        )

        self.sync_tolerance_sec = float(
            self.get_parameter('sync_tolerance_sec').value
        )

        self.depth_buffer_size = int(
            self.get_parameter('depth_buffer_size').value
        )

        self.print_interval_sec = float(
            self.get_parameter('print_interval_sec').value
        )

        # ==============================================================
        # [3] 기본 객체 / 상태
        # ==============================================================

        self.bridge = CvBridge()

        # CameraInfo는 실행 중 거의 변하지 않으므로 최신 값을 보관합니다.
        self.camera_info: Optional[CameraInfo] = None

        # Depth 버퍼 원소:
        #   (timestamp_ns, depth_image, encoding, frame_id)
        #
        # YOLO 결과는 추론 때문에 약간 늦게 도착하므로
        # Detection의 원본 영상 timestamp와 가장 가까운 Depth를
        # 이 버퍼에서 찾습니다.
        self.depth_buffer: Deque[
            Tuple[int, np.ndarray, str, str]
        ] = deque(
            maxlen=max(1, self.depth_buffer_size)
        )

        self.last_print_time = 0.0

        # ==============================================================
        # [4] QoS
        # ==============================================================

        # 현재 RealSense 영상 수신에서 안정적으로 사용했던 설정과 맞춰
        # RELIABLE / KEEP_LAST를 사용합니다.
        sensor_qos = QoSProfile(
            reliability=QoSReliabilityPolicy.RELIABLE,
            durability=QoSDurabilityPolicy.VOLATILE,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=5,
        )

        detection_qos = QoSProfile(
            reliability=QoSReliabilityPolicy.RELIABLE,
            durability=QoSDurabilityPolicy.VOLATILE,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=5,
        )

        # ==============================================================
        # [5] Subscriber
        # ==============================================================

        self.depth_subscription = self.create_subscription(
            Image,
            self.aligned_depth_topic,
            self.depth_callback,
            sensor_qos,
        )

        self.camera_info_subscription = self.create_subscription(
            CameraInfo,
            self.camera_info_topic,
            self.camera_info_callback,
            sensor_qos,
        )

        self.detection_subscription = self.create_subscription(
            Detection2DArray,
            self.detections_topic,
            self.detections_callback,
            detection_qos,
        )

        # ==============================================================
        # [6] Publisher
        # ==============================================================

        self.position_publisher = self.create_publisher(
            PoseArray,
            self.positions_topic,
            10,
        )

        # class / confidence / 3D 위치를 한 메시지에 함께 유지하기 위한
        # 표준 vision_msgs 3D Detection Publisher입니다.
        self.detection3d_publisher = self.create_publisher(
            Detection3DArray,
            self.detections_3d_topic,
            10,
        )

        # ==============================================================
        # [7] 시작 로그
        # ==============================================================

        self.get_logger().info(
            'Depth Position Node 시작'
        )

        self.get_logger().info(
            f'Detection Topic : {self.detections_topic}'
        )

        self.get_logger().info(
            f'Aligned Depth   : {self.aligned_depth_topic}'
        )

        self.get_logger().info(
            f'CameraInfo      : {self.camera_info_topic}'
        )

        self.get_logger().info(
            f'XYZ Output      : {self.positions_topic}'
        )

        self.get_logger().info(
            f'3D Detection    : {self.detections_3d_topic}'
        )

        self.get_logger().info(
            '좌표계: +X right, +Y down, +Z forward'
        )

    # ==================================================================
    # ROS Header timestamp -> nanoseconds
    # ==================================================================

    @staticmethod
    def stamp_to_ns(stamp) -> int:
        """
        builtin_interfaces/Time을 정수 nanosecond로 변환합니다.

        Detection과 Depth의 시간 차이를 계산하기 위해 사용합니다.
        """

        return (
            int(stamp.sec) * 1_000_000_000
            + int(stamp.nanosec)
        )

    # ==================================================================
    # CameraInfo Callback
    # ==================================================================

    def camera_info_callback(
        self,
        msg: CameraInfo,
    ) -> None:
        """
        카메라 내부 파라미터를 저장합니다.

        CameraInfo.k 배열:
            [fx,  0, cx,
              0, fy, cy,
              0,  0,  1]
        """

        self.camera_info = msg

    # ==================================================================
    # Depth Callback
    # ==================================================================

    def depth_callback(
        self,
        msg: Image,
    ) -> None:
        """
        aligned depth 영상을 NumPy 배열로 변환해 최근 프레임 버퍼에 저장합니다.

        중요:
        이 Topic은 Color 좌표계로 정렬된 Depth여야 합니다.
        그래야 YOLO에서 얻은 동일한 (u, v)를 그대로 사용할 수 있습니다.
        """

        try:
            depth_image = self.bridge.imgmsg_to_cv2(
                msg,
                desired_encoding='passthrough',
            )

        except CvBridgeError as error:
            self.get_logger().error(
                f'Depth cv_bridge 변환 실패: {error}'
            )
            return

        timestamp_ns = self.stamp_to_ns(
            msg.header.stamp
        )

        # Callback이 끝난 뒤에도 안전하게 보관하기 위해 복사합니다.
        self.depth_buffer.append(
            (
                timestamp_ns,
                depth_image.copy(),
                msg.encoding,
                msg.header.frame_id,
            )
        )

    # ==================================================================
    # Detection timestamp와 가장 가까운 Depth 검색
    # ==================================================================

    def find_nearest_depth(
        self,
        detection_timestamp_ns: int,
    ):
        """
        YOLO Detection의 원본 RGB timestamp와 가장 가까운 Depth를 찾습니다.

        YOLO는 추론 시간이 있기 때문에 Detection Topic이 실제 RGB보다
        약간 늦게 publish될 수 있습니다.

        따라서 "현재 최신 Depth"를 무조건 쓰지 않고,
        Detection2DArray.header.stamp와 가장 가까운 프레임을 찾습니다.
        """

        if not self.depth_buffer:
            return None

        nearest = min(
            self.depth_buffer,
            key=lambda item: abs(
                item[0] - detection_timestamp_ns
            ),
        )

        time_difference_sec = abs(
            nearest[0] - detection_timestamp_ns
        ) / 1_000_000_000.0

        if (
            time_difference_sec
            > self.sync_tolerance_sec
        ):
            self.get_logger().warning(
                'Detection과 Depth 시간 차이가 너무 큽니다: '
                f'{time_difference_sec:.3f}초 '
                f'(허용 {self.sync_tolerance_sec:.3f}초)'
            )
            return None

        return nearest

    # ==================================================================
    # 안정적인 Depth 중앙값 계산
    # ==================================================================

    def robust_depth(
        self,
        depth_image: np.ndarray,
        encoding: str,
        u: int,
        v: int,
    ) -> Optional[float]:
        """
        중심점 주변 ROI의 유효 Depth 중앙값을 meter 단위로 반환합니다.

        원본 camera_detection 코드의 robust_depth()와 같은 목적입니다.

        16UC1:
            depth_scale을 곱해 meter로 변환합니다.

        32FC1:
            일반적으로 값 자체를 meter로 사용합니다.

        중심점 한 픽셀만 사용하지 않는 이유:
            - 반사에 의한 0 Depth
            - 튀는 Depth
            - Bounding Box 중심이 빈 공간에 걸리는 경우
        """

        if depth_image.ndim != 2:
            self.get_logger().warning(
                f'예상하지 못한 Depth shape: {depth_image.shape}'
            )
            return None

        height, width = depth_image.shape[:2]

        # 영상 밖 좌표 방지
        if (
            u < 0
            or u >= width
            or v < 0
            or v >= height
        ):
            return None

        x_start = max(
            0,
            u - self.depth_radius,
        )

        x_end = min(
            width,
            u + self.depth_radius + 1,
        )

        y_start = max(
            0,
            v - self.depth_radius,
        )

        y_end = min(
            height,
            v + self.depth_radius + 1,
        )

        roi = depth_image[
            y_start:y_end,
            x_start:x_end,
        ].astype(
            np.float32,
            copy=False,
        )

        normalized_encoding = (
            encoding.strip().upper()
        )

        # RealSense Z16 / ROS 16UC1 계열
        if (
            normalized_encoding == '16UC1'
            or normalized_encoding == 'MONO16'
            or depth_image.dtype == np.uint16
        ):
            depth_m = (
                roi
                * self.depth_scale
            )

        # float depth는 일반적으로 meter 단위입니다.
        elif (
            normalized_encoding == '32FC1'
            or np.issubdtype(
                depth_image.dtype,
                np.floating,
            )
        ):
            depth_m = roi

        else:
            self.get_logger().warning(
                '지원하지 않는 Depth encoding: '
                f'{encoding}, dtype={depth_image.dtype}'
            )
            return None

        valid_mask = (
            np.isfinite(depth_m)
            & (depth_m >= self.min_depth_m)
            & (depth_m <= self.max_depth_m)
        )

        valid_values = depth_m[
            valid_mask
        ]

        if valid_values.size == 0:
            return None

        return float(
            np.median(
                valid_values
            )
        )

    # ==================================================================
    # Pixel -> Camera XYZ
    # ==================================================================

    def deproject(
        self,
        u: float,
        v: float,
        depth_m: float,
    ) -> Optional[
        Tuple[float, float, float]
    ]:
        """
        CameraInfo의 pinhole camera intrinsic을 이용해
        픽셀 (u, v) + Depth Z를 Camera optical frame (X, Y, Z)로 변환합니다.

        공식:
            X = (u - cx) * Z / fx
            Y = (v - cy) * Z / fy
            Z = depth

        좌표 방향:
            +X = 영상 오른쪽
            +Y = 영상 아래쪽
            +Z = 카메라 전방
        """

        if self.camera_info is None:
            return None

        k = self.camera_info.k

        fx = float(k[0])
        fy = float(k[4])
        cx = float(k[2])
        cy = float(k[5])

        if (
            abs(fx) < 1e-9
            or abs(fy) < 1e-9
        ):
            self.get_logger().error(
                'CameraInfo의 fx 또는 fy가 0입니다.'
            )
            return None

        x_m = (
            (float(u) - cx)
            * depth_m
            / fx
        )

        y_m = (
            (float(v) - cy)
            * depth_m
            / fy
        )

        z_m = depth_m

        return (
            float(x_m),
            float(y_m),
            float(z_m),
        )

    # ==================================================================
    # YOLO Detection Callback
    # ==================================================================

    def detections_callback(
        self,
        msg: Detection2DArray,
    ) -> None:
        """
        YOLO 2D Detection 결과를 3D Camera 좌표로 변환합니다.

        처리 순서:
            Detection2DArray
                ↓
            bbox center (u, v)
                ↓
            timestamp가 가장 가까운 aligned depth 선택
                ↓
            중심 주변 Depth 중앙값
                ↓
            CameraInfo intrinsic
                ↓
            X, Y, Z
                ↓
            PoseArray publish
        """

        if self.camera_info is None:
            self.get_logger().warning(
                '아직 CameraInfo를 수신하지 못했습니다.'
            )
            return

        if not msg.detections:
            return

        detection_timestamp_ns = (
            self.stamp_to_ns(
                msg.header.stamp
            )
        )

        nearest_depth = self.find_nearest_depth(
            detection_timestamp_ns
        )

        if nearest_depth is None:
            return

        (
            _,
            depth_image,
            depth_encoding,
            depth_frame_id,
        ) = nearest_depth

        output = PoseArray()
        output_3d = Detection3DArray()

        # aligned depth가 color frame으로 publish되므로
        # depth header의 optical frame을 XYZ 출력 frame으로 사용합니다.
        output.header = copy.deepcopy(msg.header)
        output_3d.header = copy.deepcopy(msg.header)

        if depth_frame_id:
            output.header.frame_id = depth_frame_id
            output_3d.header.frame_id = depth_frame_id

        terminal_messages = []

        for object_index, detection in enumerate(
            msg.detections,
            start=1,
        ):
            # ----------------------------------------------------------
            # Bounding Box 중심 = YOLO 객체 중심 Pixel
            # ----------------------------------------------------------

            u_float = float(
                detection
                .bbox
                .center
                .position
                .x
            )

            v_float = float(
                detection
                .bbox
                .center
                .position
                .y
            )

            u = int(
                round(
                    u_float
                )
            )

            v = int(
                round(
                    v_float
                )
            )

            # ----------------------------------------------------------
            # 클래스 / Confidence
            # ----------------------------------------------------------

            class_id = 'unknown'
            confidence = math.nan

            if detection.results:
                best_result = max(
                    detection.results,
                    key=lambda result: (
                        result.hypothesis.score
                    ),
                )

                class_id = (
                    best_result
                    .hypothesis
                    .class_id
                )

                confidence = float(
                    best_result
                    .hypothesis
                    .score
                )

            # ----------------------------------------------------------
            # Depth
            # ----------------------------------------------------------

            depth_m = self.robust_depth(
                depth_image,
                depth_encoding,
                u,
                v,
            )

            if depth_m is None:
                terminal_messages.append(
                    f'객체 {object_index}: '
                    f'class={class_id}, '
                    f'uv=({u},{v}), '
                    'XYZ=unavailable'
                )
                continue

            # ----------------------------------------------------------
            # Pixel -> Camera XYZ
            # ----------------------------------------------------------

            xyz = self.deproject(
                u_float,
                v_float,
                depth_m,
            )

            if xyz is None:
                continue

            x_m, y_m, z_m = xyz

            # ----------------------------------------------------------
            # ROS Pose 생성
            # ----------------------------------------------------------
            #
            # 여기서는 객체의 "위치"만 알고 있고 자세는 계산하지 않으므로
            # orientation은 단위 quaternion으로 둡니다.
            pose = Pose()

            pose.position.x = x_m
            pose.position.y = y_m
            pose.position.z = z_m

            pose.orientation.x = 0.0
            pose.orientation.y = 0.0
            pose.orientation.z = 0.0
            pose.orientation.w = 1.0

            output.poses.append(
                pose
            )

            # ----------------------------------------------------------
            # class / confidence까지 보존하는 Detection3D 생성
            # ----------------------------------------------------------
            detection_3d = Detection3D()
            detection_3d.header = copy.deepcopy(output_3d.header)

            detection_3d.bbox.center.position.x = x_m
            detection_3d.bbox.center.position.y = y_m
            detection_3d.bbox.center.position.z = z_m
            detection_3d.bbox.center.orientation.w = 1.0

            # 현재 단계에서는 객체의 실제 3차원 폭/높이/깊이를 계산하지 않으므로
            # bbox.size는 0으로 둡니다. 여기서는 "3D 중심 위치"가 핵심입니다.
            detection_3d.bbox.size.x = 0.0
            detection_3d.bbox.size.y = 0.0
            detection_3d.bbox.size.z = 0.0

            if detection.results:
                best_3d_result = copy.deepcopy(
                    best_result
                )

                # 2D Detection에 있던 class / score를 유지하면서
                # 해당 hypothesis의 pose에도 계산된 XYZ를 기록합니다.
                best_3d_result.pose.pose.position.x = x_m
                best_3d_result.pose.pose.position.y = y_m
                best_3d_result.pose.pose.position.z = z_m
                best_3d_result.pose.pose.orientation.w = 1.0

                detection_3d.results.append(
                    best_3d_result
                )

            output_3d.detections.append(
                detection_3d
            )

            # ----------------------------------------------------------
            # 터미널 표시
            # ----------------------------------------------------------

            if math.isnan(confidence):
                confidence_text = 'N/A'
            else:
                confidence_text = (
                    f'{confidence:.3f}'
                )

            terminal_messages.append(
                f'객체 {object_index}: '
                f'class={class_id}, '
                f'conf={confidence_text}, '
                f'uv=({u},{v}), '
                f'XYZ=('
                f'{x_m:+.3f},'
                f'{y_m:+.3f},'
                f'{z_m:.3f}'
                f') m'
            )

        # 유효한 XYZ가 하나 이상 있을 때만 publish합니다.
        if output.poses:
            self.position_publisher.publish(
                output
            )

        if output_3d.detections:
            self.detection3d_publisher.publish(
                output_3d
            )

        # 너무 빠르게 터미널을 도배하지 않도록 출력 주기를 제한합니다.
        now = time.perf_counter()

        if (
            terminal_messages
            and (
                now - self.last_print_time
                >= self.print_interval_sec
            )
        ):
            self.get_logger().info(
                ' | '.join(
                    terminal_messages
                )
            )

            self.last_print_time = now


def main(args=None) -> None:
    """
    ROS 2 노드를 실행합니다.
    """

    rclpy.init(
        args=args
    )

    node = DepthPositionNode()

    try:
        rclpy.spin(
            node
        )

    except KeyboardInterrupt:
        pass

    finally:
        node.destroy_node()

        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()