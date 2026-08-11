#!/usr/bin/env python3

import threading
import time
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import rclpy
import torch

from cv_bridge import CvBridge, CvBridgeError
from rclpy.node import Node
from rclpy.qos import (
    QoSProfile,
    QoSReliabilityPolicy,
    QoSDurabilityPolicy,
    QoSHistoryPolicy,
)
from sensor_msgs.msg import Image
from ultralytics import YOLO


import copy

from vision_msgs.msg import (
    Detection2D,
    Detection2DArray,
    ObjectHypothesisWithPose,
)


class YoloDetectionNode(Node):
    """
    RealSense RGB ROS Topic을 구독하여 YOLO 추론을 수행합니다.

    =====================================================================
    전체 구조
    =====================================================================

    [Python Main Thread]

        OpenCV HighGUI 사전 초기화
        namedWindow()
        imshow(dummy)
        waitKey()
              │
              │ 반드시 YOLO predict()보다 먼저 수행
              ▼

        rclpy.spin_once()
              │
              ├── image_callback()
              ├── report_statistics()
              │
              ▼
        display_once()
              │
              ├── cv2.imshow()
              └── cv2.waitKey()


    [YOLO Worker Thread]

        latest_frame 확인
              │
              ▼
        model.predict()
              │
              ▼
        results[0].plot()
              │
              ▼
        latest_display_frame 저장


    =====================================================================
    이 구조를 사용하는 이유
    =====================================================================

    실험 결과 현재 환경에서는:

        OpenCV imshow만 실행                  → 정상
        Ultralytics import + imshow          → 정상
        PyTorch CUDA + imshow                → 정상
        YOLO predict 후 처음 imshow          → 정지

    하지만:

        OpenCV HighGUI 먼저 초기화
        → YOLO predict
        → imshow

    순서에서는 정상 동작했습니다.

    따라서 이 코드에서는 YOLO Worker를 시작하기 전에
    OpenCV Window를 실제 Main Thread에서 먼저 생성하고
    GUI Event까지 한 번 처리합니다.
    """

    def __init__(self) -> None:
        super().__init__('yolo_detection_node')

        # ==============================================================
        # [1] ROS 파라미터 선언
        # ==============================================================

        # RealSense RGB Image Topic
        self.declare_parameter(
            'image_topic',
            '/camera/camera/color/image_raw'
        )

        # 학습된 YOLO 모델
        self.declare_parameter(
            'model_path',
            str(
                Path.home()
                / 'ria'
                / 'RIA_food_ws'
                / 'models'
                / 'batter_done_best.pt'
            )
        )

        # YOLO Confidence Threshold
        self.declare_parameter(
            'confidence',
            0.50
        )

        # YOLO 입력 이미지 크기
        self.declare_parameter(
            'image_size',
            640
        )

        # 결과 화면 표시 여부
        self.declare_parameter(
            'show_window',
            True
        )

        # 추론 장치
        #
        # "0"    : NVIDIA GPU 0
        # "cpu"  : CPU
        # "auto" : CUDA 가능 여부에 따라 자동 선택
        self.declare_parameter(
            'device',
            '0'
        )

        # 성능 통계 출력 주기 [초]
        self.declare_parameter(
            'report_interval',
            1.0
        )

        # 이 시간 이상 이미지가 들어오지 않으면
        # 수신 정지로 판단합니다.
        self.declare_parameter(
            'stall_threshold',
            0.5
        )

        # ==============================================================
        # [2] ROS 파라미터 읽기
        # ==============================================================

        self.image_topic = self._get_string_parameter(
            'image_topic'
        )

        self.model_path = self._get_string_parameter(
            'model_path'
        )

        self.confidence = self._get_double_parameter(
            'confidence'
        )

        self.image_size = self._get_integer_parameter(
            'image_size'
        )

        self.show_window = self._get_bool_parameter(
            'show_window'
        )

        self.device_parameter = self._get_string_parameter(
            'device'
        )

        self.report_interval = self._get_double_parameter(
            'report_interval'
        )

        self.stall_threshold = self._get_double_parameter(
            'stall_threshold'
        )

        # ==============================================================
        # [3] 모델 경로 확인
        # ==============================================================

        model_file = Path(
            self.model_path
        ).expanduser()

        if not model_file.is_file():
            raise FileNotFoundError(
                f'YOLO 모델을 찾을 수 없습니다: {model_file}'
            )

        self.model_path = str(
            model_file
        )

        # ==============================================================
        # [4] 추론 장치 설정
        # ==============================================================

        self.device = self._select_device(
            self.device_parameter
        )

        # ==============================================================
        # [5] OpenCV / ROS Image 변환 객체
        # ==============================================================

        self.bridge = CvBridge()

        # OpenCV Window 이름은 생성부터 종료까지
        # 동일한 문자열을 사용합니다.
        self.window_name = 'Fried Food YOLO ROS2'

        # ==============================================================
        # [6] 중요: OpenCV HighGUI 사전 초기화
        # ==============================================================

        # 지금까지의 테스트에서 가장 중요한 부분입니다.
        #
        # YOLO의 model.predict()가 처음 실행된 뒤에
        # cv2.imshow()를 최초 호출하면 현재 환경에서 정지했습니다.
        #
        # 따라서 YOLO 모델 추론이 시작되기 전에
        #
        #   namedWindow()
        #       ↓
        #   imshow(dummy)
        #       ↓
        #   waitKey()
        #
        # 순서로 HighGUI를 먼저 완전히 초기화합니다.
        #
        # __init__()은 Python Main Thread에서 호출되기 때문에
        # 이 지점에서 GUI 초기화를 수행하는 것이 중요합니다.
        if self.show_window:
            self._initialize_gui()

        # ==============================================================
        # [7] YOLO 모델 로딩
        # ==============================================================

        # GUI 초기화를 먼저 한 다음 YOLO 모델을 준비합니다.
        #
        # 모델 로딩 자체는 이전 테스트에서도 문제가 없었지만,
        # 정상 동작했던 최소 테스트의 순서와 동일하게 맞추기 위해
        # HighGUI 초기화 이후에 모델을 로드합니다.
        self.get_logger().info(
            f'YOLO 모델 로딩: {self.model_path}'
        )

        self.get_logger().info(
            f'추론 장치: {self.device}'
        )

        self.model = YOLO(
            self.model_path
        )

        # ==============================================================
        # [8] 입력 프레임 공유 변수
        # ==============================================================

        # ROS callback과 YOLO Worker Thread가
        # latest_frame을 동시에 접근할 수 있으므로 Lock을 사용합니다.
        self.frame_lock = threading.Lock()

        # 가장 최근에 수신한 RGB 프레임
        self.latest_frame: Optional[np.ndarray] = None

        # 가장 최근 프레임의 번호
        self.latest_frame_sequence = 0

        # YOLO가 마지막으로 처리한 프레임 번호
        self.last_processed_sequence = 0

        # ==============================================================
        # [9] 결과 이미지 공유 변수
        # ==============================================================

        # YOLO Worker가 결과를 저장하고
        # Main Thread가 화면에 출력하므로
        # 별도의 Lock을 사용합니다.
        self.display_lock = threading.Lock()

        self.latest_display_frame: Optional[np.ndarray] = None

        # ==============================================================
        # [10] 실행 상태 및 Worker Thread
        # ==============================================================

        self.running = True

        self.worker_thread = threading.Thread(
            target=self.inference_loop,
            name='yolo_inference_worker',
            daemon=True
        )

        # ==============================================================
        # [11] ROS 이미지 수신 통계
        # ==============================================================

        now = time.perf_counter()

        self.start_time = now
        self.last_report_time = now

        self.last_receive_time: Optional[float] = None

        self.total_received_frames = 0
        self.interval_received_frames = 0

        self.receive_stall_count = 0
        self.max_receive_interval = 0.0

        # ==============================================================
        # [12] YOLO 추론 통계
        # ==============================================================

        self.total_inference_frames = 0
        self.interval_inference_frames = 0

        self.total_inference_time = 0.0
        self.interval_inference_time = 0.0

        # YOLO가 이전 프레임을 처리하기 전에
        # 새로운 프레임으로 교체된 횟수
        self.replaced_frame_count = 0

        # ==============================================================
        # [13] RealSense Image QoS
        # ==============================================================

        # 확인된 RealSense Publisher와 동일한 조건입니다.
        #
        # Reliability : RELIABLE
        # Durability  : VOLATILE
        # History     : KEEP_LAST
        # Depth       : 1
        #
        # 오래된 프레임을 누적하지 않고
        # 최신 영상 중심으로 처리합니다.
        image_qos = QoSProfile(
            reliability=QoSReliabilityPolicy.RELIABLE,
            durability=QoSDurabilityPolicy.VOLATILE,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1,
        )

        # ==============================================================
        # [14] RealSense RGB Subscriber
        # ==============================================================

        self.subscription = self.create_subscription(
            Image,
            self.image_topic,
            self.image_callback,
            image_qos
        )

        # ==============================================================
        # [15] 성능 통계 Timer
        # ==============================================================

        # 화면 출력용 Timer가 아닙니다.
        #
        # 1초마다 FPS 등의 성능 통계만 출력합니다.
        self.report_timer = self.create_timer(
            self.report_interval,
            self.report_statistics
        )

        # ==============================================================
        # [16] YOLO Worker Thread 시작
        # ==============================================================

        # 중요:
        #
        # 반드시 _initialize_gui()보다 나중에 실행되어야 합니다.
        #
        # 이 Worker가 시작된 이후 실제 model.predict()가
        # 수행되기 때문입니다.
        self.worker_thread.start()

        # ==============================================================
        # [17] 시작 로그
        # ==============================================================

        self.get_logger().info(
            f'이미지 구독 시작: {self.image_topic}'
        )

        self.get_logger().info(
            'RELIABLE / KEEP_LAST(1) QoS와 '
            '최신 프레임 처리 구조를 사용합니다.'
        )

        if self.show_window:
            self.get_logger().info(
                'OpenCV HighGUI를 YOLO 추론 전에 '
                '사전 초기화했습니다.'
            )

            self.get_logger().info(
                'OpenCV 화면 출력은 Python Main Thread에서 처리합니다.'
            )
        # ==============================================================
        # YOLO 2D Detection 결과 Publisher
        # ==============================================================

        # Bounding Box, 중심 픽셀(u, v), class, confidence를
        # 다른 ROS 노드에서 사용할 수 있도록 publish합니다.
        self.detection_publisher = self.create_publisher(
            Detection2DArray,
            '/fried_food_vision/detections',
            10,
        )

        # YOLO Worker가 처리하는 프레임이 어느 RGB 영상에서 왔는지
        # timestamp를 유지하기 위해 Header도 같이 저장합니다.
        self.latest_header = None 

    # ==================================================================
    # OpenCV HighGUI 초기화
    # ==================================================================

    def _initialize_gui(self) -> None:
        """
        YOLO 추론 전에 OpenCV HighGUI를 먼저 초기화합니다.

        현재 환경에서 확인된 현상:

            YOLO predict
                ↓
            최초 imshow
                ↓
            정지

        반면:

            namedWindow
                ↓
            dummy imshow
                ↓
            waitKey
                ↓
            YOLO predict
                ↓
            imshow
                ↓
            정상

        이므로 정상 동작이 확인된 초기화 순서를 그대로 적용합니다.
        """

        self.get_logger().info(
            'OpenCV HighGUI 사전 초기화를 시작합니다.'
        )

        # Window를 명시적으로 먼저 생성합니다.
        cv2.namedWindow(
            self.window_name,
            cv2.WINDOW_NORMAL
        )

        # 필요하면 화면 크기를 조정할 수 있습니다.
        cv2.resizeWindow(
            self.window_name,
            960,
            720
        )

        # 아직 RealSense/YOLO 결과가 없으므로
        # GUI Backend 초기화용 빈 이미지를 생성합니다.
        dummy_frame = np.zeros(
            (480, 640, 3),
            dtype=np.uint8
        )

        cv2.putText(
            dummy_frame,
            'Waiting for YOLO...',
            (170, 240),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2,
            cv2.LINE_AA
        )

        # 실제 Window에 이미지를 한 번 전달합니다.
        cv2.imshow(
            self.window_name,
            dummy_frame
        )

        # 단순히 Window만 만드는 것으로 끝내지 않고
        # GUI Event까지 실제로 한 번 처리합니다.
        #
        # 최소 테스트에서 정상 동작했던 조건과 동일하게
        # 100 ms 동안 Event 처리를 수행합니다.
        cv2.waitKey(100)

        self.get_logger().info(
            'OpenCV HighGUI 사전 초기화 완료'
        )

    # ==================================================================
    # ROS 파라미터 읽기 함수
    # ==================================================================

    def _get_string_parameter(
        self,
        name: str
    ) -> str:

        return (
            self.get_parameter(name)
            .get_parameter_value()
            .string_value
        )

    def _get_double_parameter(
        self,
        name: str
    ) -> float:

        return (
            self.get_parameter(name)
            .get_parameter_value()
            .double_value
        )

    def _get_integer_parameter(
        self,
        name: str
    ) -> int:

        return (
            self.get_parameter(name)
            .get_parameter_value()
            .integer_value
        )

    def _get_bool_parameter(
        self,
        name: str
    ) -> bool:

        return (
            self.get_parameter(name)
            .get_parameter_value()
            .bool_value
        )

    # ==================================================================
    # YOLO 추론 장치 선택
    # ==================================================================

    def _select_device(
        self,
        requested_device: str
    ) -> str:
        """
        YOLO 추론 장치를 결정합니다.

        "auto":
            CUDA 사용 가능 → GPU 0
            CUDA 사용 불가 → CPU

        "0":
            NVIDIA GPU 0

        "cpu":
            CPU
        """

        normalized = (
            requested_device
            .strip()
            .lower()
        )

        if normalized == 'auto':
            return (
                '0'
                if torch.cuda.is_available()
                else 'cpu'
            )

        if (
            normalized != 'cpu'
            and not torch.cuda.is_available()
        ):
            self.get_logger().warning(
                'CUDA를 사용할 수 없어 CPU로 변경합니다.'
            )

            return 'cpu'

        return requested_device

    # ==================================================================
    # ROS Image Callback
    # ==================================================================

    def image_callback(
        self,
        msg: Image
    ) -> None:
        """
        RealSense RGB Image를 수신합니다.

        이 Callback에서는 YOLO 추론을 수행하지 않습니다.

        처리 순서:

            ROS Image
                ↓
            cv_bridge
                ↓
            BGR NumPy Image
                ↓
            latest_frame 저장

        YOLO를 Callback 내부에서 직접 실행하지 않는 이유는
        YOLO 연산 때문에 다음 ROS Image 수신이 지연되는 것을
        방지하기 위해서입니다.
        """

        receive_time = time.perf_counter()

        # --------------------------------------------------------------
        # 프레임 수신 간격 측정
        # --------------------------------------------------------------

        if self.last_receive_time is not None:

            receive_interval = (
                receive_time
                - self.last_receive_time
            )

            self.max_receive_interval = max(
                self.max_receive_interval,
                receive_interval
            )

            if (
                receive_interval
                >= self.stall_threshold
            ):

                self.receive_stall_count += 1

                self.get_logger().warning(
                    '이미지 수신 정지 감지: '
                    f'{receive_interval:.3f}초'
                )

        self.last_receive_time = receive_time

        # --------------------------------------------------------------
        # 입력 FPS 통계
        # --------------------------------------------------------------

        self.total_received_frames += 1
        self.interval_received_frames += 1

        # --------------------------------------------------------------
        # ROS Image → OpenCV BGR
        # --------------------------------------------------------------

        try:

            frame = self.bridge.imgmsg_to_cv2(
                msg,
                desired_encoding='bgr8'
            )

        except CvBridgeError as error:

            self.get_logger().error(
                f'cv_bridge 변환 실패: {error}'
            )

            return

        # --------------------------------------------------------------
        # 최신 프레임 저장
        # --------------------------------------------------------------

        with self.frame_lock:

            # 아직 이전 최신 프레임을 YOLO가 처리하지 못했는데
            # 새로운 프레임이 도착한 경우입니다.
            #
            # 과거 프레임을 큐에 계속 쌓지 않고
            # 새로운 영상으로 교체합니다.
            if (
                self.latest_frame_sequence
                > self.last_processed_sequence
            ):
                self.replaced_frame_count += 1

            # ROS Message 메모리와 분리하기 위해 복사합니다.
            self.latest_frame = frame.copy()
            #----------------------------------------------------------
            # 원본 RGB 영상의 timestamp / frame_id도 함께 보존
            #----------------------------------------------------------
            #
            # 나중에 Depth 영상과 시간적으로 같은 프레임을 찾기 위해
            # 매우 중요한 정보입니다          
            self.latest_header = copy.deepcopy(
                msg.header
            )           
            self.latest_frame_sequence += 1

    # ==================================================================
    # YOLO Worker Thread
    # ==================================================================

    def inference_loop(self) -> None:
        """
        최신 RGB Image에 대해 YOLO 추론을 반복합니다.

        이 함수는 별도의 Worker Thread에서 실행됩니다.

        여기서는 절대로:

            cv2.imshow()
            cv2.waitKey()

        를 실행하지 않습니다.

        역할은 다음과 같습니다.

            최신 프레임 획득
                ↓
            YOLO predict
                ↓
            검출 결과 이미지 생성
                ↓
            latest_display_frame 저장
        """

        while (
            self.running
            and rclpy.ok()
        ):

            frame = None
            frame_header = None
            sequence = 0

            # ----------------------------------------------------------
            # 최신 Image 가져오기
            # ----------------------------------------------------------

            with self.frame_lock:

                if (
                    self.latest_frame is not None
                    and
                    self.latest_frame_sequence
                    != self.last_processed_sequence
                ):

                    frame = self.latest_frame.copy()

                    frame_header = copy.deepcopy(
                        self.latest_header
                    )

                    sequence = (
                        self.latest_frame_sequence
                    )

            # 새로운 프레임이 아직 없다면
            # CPU Busy Waiting을 줄이기 위해 잠시 대기합니다.
            if frame is None:

                time.sleep(0.001)

                continue

            # ----------------------------------------------------------
            # YOLO 추론
            # ----------------------------------------------------------

            inference_start = (
                time.perf_counter()
            )

            try:

                results = self.model.predict(
                    source=frame,
                    conf=self.confidence,
                    imgsz=self.image_size,
                    device=self.device,
                    verbose=False
                )
                # ==========================================================
                # YOLO 검출 결과를 다른 ROS 노드에 publish
                # ==========================================================

                if results:

                    self.publish_detections(
                        results[0],
                        frame_header,
                    )

            except Exception as error:

                self.get_logger().error(
                    f'YOLO 추론 실패: {error}'
                )

                time.sleep(0.1)

                continue

            # ----------------------------------------------------------
            # 추론 시간
            # ----------------------------------------------------------

            inference_time = (
                time.perf_counter()
                - inference_start
            )

            # ----------------------------------------------------------
            # YOLO 성능 통계
            # ----------------------------------------------------------

            self.total_inference_frames += 1

            self.interval_inference_frames += 1

            self.total_inference_time += (
                inference_time
            )

            self.interval_inference_time += (
                inference_time
            )

            # 해당 Sequence까지 처리 완료
            self.last_processed_sequence = sequence

            # ----------------------------------------------------------
            # YOLO 결과 이미지 생성
            # ----------------------------------------------------------

            if (
                self.show_window
                and results
            ):

                # YOLO Bounding Box, Class, Confidence 등이
                # 그려진 BGR 이미지를 생성합니다.
                annotated_frame = (
                    results[0].plot()
                )

                # ------------------------------------------------------
                # 한 프레임의 YOLO 계산 시간 기준 FPS
                # ------------------------------------------------------
                #
                # 주의:
                # 이 값은 Camera 전체 처리 FPS가 아닙니다.
                #
                # 예:
                # YOLO 연산 10 ms → 약 100 FPS 계산 능력
                #
                # 그러나 Camera가 15 FPS로 들어오면
                # 실제 처리 FPS는 약 15 FPS가 됩니다.
                instant_inference_fps = (
                    1.0 / inference_time
                    if inference_time > 0.0
                    else 0.0
                )

                cv2.putText(
                    annotated_frame,
                    f'YOLO compute: {instant_inference_fps:.1f} FPS',
                    (20, 35),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 0),
                    2,
                    cv2.LINE_AA
                )

                # ------------------------------------------------------
                # 결과 Image만 저장
                # ------------------------------------------------------
                #
                # GUI는 Main Thread에서만 처리합니다.
                with self.display_lock:

                    self.latest_display_frame = (
                        annotated_frame.copy()
                    )

    # ==================================================================
    # OpenCV 화면 출력
    # ==================================================================

    def display_once(self) -> None:
        """
        최신 YOLO 결과를 화면에 한 번 표시합니다.

        Timer Callback이 아닙니다.

        main()의 while Loop에서 직접 호출되므로
        실제 Python Main Thread에서 실행됩니다.
        """

        if not self.show_window:
            return

        display_frame = None

        # --------------------------------------------------------------
        # 최신 YOLO 결과 가져오기
        # --------------------------------------------------------------

        with self.display_lock:

            if (
                self.latest_display_frame
                is not None
            ):

                display_frame = (
                    self.latest_display_frame.copy()
                )

        # 아직 YOLO 첫 추론이 끝나지 않은 경우
        # 초기화용 Waiting 화면을 그대로 유지합니다.
        if display_frame is None:

            # GUI Event는 계속 처리해야 하므로
            # waitKey는 호출합니다.
            key = cv2.waitKey(1) & 0xFF

            if key in (
                27,
                ord('q')
            ):
                self.running = False

            return

        # --------------------------------------------------------------
        # 최신 YOLO Image 표시
        # --------------------------------------------------------------

        cv2.imshow(
            self.window_name,
            display_frame
        )

        # --------------------------------------------------------------
        # OpenCV GUI Event 처리
        # --------------------------------------------------------------

        key = cv2.waitKey(1) & 0xFF

        # --------------------------------------------------------------
        # q 또는 ESC 입력 시 종료
        # --------------------------------------------------------------

        if key in (
            27,
            ord('q')
        ):

            self.get_logger().info(
                '화면 종료 요청을 받았습니다.'
            )

            self.running = False

            return

        # --------------------------------------------------------------
        # X 버튼으로 Window를 닫은 경우
        # --------------------------------------------------------------

        try:

            visible = cv2.getWindowProperty(
                self.window_name,
                cv2.WND_PROP_VISIBLE
            )

            if visible < 1:

                self.get_logger().info(
                    'OpenCV Window가 닫혔습니다.'
                )

                self.running = False

        except cv2.error:

            # Window가 이미 완전히 제거된 경우
            # getWindowProperty()가 예외를 발생시킬 수 있습니다.
            self.running = False

    # ==================================================================
    # 성능 통계
    # ==================================================================

    def report_statistics(self) -> None:
        """
        report_interval마다 다음 정보를 출력합니다.

        - ROS 입력 FPS
        - YOLO 추론 FPS
        - 평균 YOLO 추론 시간
        - 누적 FPS
        - 수신 프레임
        - 추론 프레임
        - 교체된 대기 프레임
        - 최대 수신 간격
        """

        current_time = (
            time.perf_counter()
        )

        interval_elapsed = (
            current_time
            - self.last_report_time
        )

        total_elapsed = (
            current_time
            - self.start_time
        )

        # --------------------------------------------------------------
        # ROS 입력 FPS
        # --------------------------------------------------------------

        receive_fps = (
            self.interval_received_frames
            / interval_elapsed
            if interval_elapsed > 0.0
            else 0.0
        )

        # --------------------------------------------------------------
        # 실제 YOLO 처리 FPS
        # --------------------------------------------------------------

        inference_fps = (
            self.interval_inference_frames
            / interval_elapsed
            if interval_elapsed > 0.0
            else 0.0
        )

        # --------------------------------------------------------------
        # 누적 ROS 입력 FPS
        # --------------------------------------------------------------

        cumulative_receive_fps = (
            self.total_received_frames
            / total_elapsed
            if total_elapsed > 0.0
            else 0.0
        )

        # --------------------------------------------------------------
        # 누적 YOLO FPS
        # --------------------------------------------------------------

        cumulative_inference_fps = (
            self.total_inference_frames
            / total_elapsed
            if total_elapsed > 0.0
            else 0.0
        )

        # --------------------------------------------------------------
        # 구간 평균 YOLO 추론 시간
        # --------------------------------------------------------------

        average_inference_ms = (
            self.interval_inference_time
            / self.interval_inference_frames
            * 1000.0
            if self.interval_inference_frames > 0
            else 0.0
        )

        # --------------------------------------------------------------
        # 마지막 Image 수신 후 경과 시간
        # --------------------------------------------------------------

        if self.last_receive_time is None:

            last_frame_age = float('inf')

        else:

            last_frame_age = (
                current_time
                - self.last_receive_time
            )

        # --------------------------------------------------------------
        # 결과 출력
        # --------------------------------------------------------------

        self.get_logger().info(
            '\n'
            f'ROS 입력 FPS       : {receive_fps:6.2f}\n'
            f'YOLO 추론 FPS      : {inference_fps:6.2f}\n'
            f'평균 추론 시간     : {average_inference_ms:7.2f} ms\n'
            f'누적 입력 FPS      : {cumulative_receive_fps:6.2f}\n'
            f'누적 추론 FPS      : {cumulative_inference_fps:6.2f}\n'
            f'수신 프레임        : {self.total_received_frames}\n'
            f'추론 프레임        : {self.total_inference_frames}\n'
            f'교체된 대기 프레임 : {self.replaced_frame_count}\n'
            f'최대 수신 간격     : {self.max_receive_interval:.3f}초\n'
            f'수신 정지 횟수     : {self.receive_stall_count}\n'
            f'마지막 프레임      : {last_frame_age:.3f}초 전'
        )

        # --------------------------------------------------------------
        # 다음 구간 측정을 위해 초기화
        # --------------------------------------------------------------

        self.interval_received_frames = 0

        self.interval_inference_frames = 0

        self.interval_inference_time = 0.0

        self.last_report_time = current_time

    # ==================================================================
    # ROS Node 종료
    # ==================================================================

    def destroy_node(self) -> bool:
        """
        종료 절차:

        1. YOLO Worker 종료 요청
        2. Worker Thread 종료 대기
        3. OpenCV Window 정리
        4. ROS Node 제거
        """

        self.running = False

        # --------------------------------------------------------------
        # Worker 종료
        # --------------------------------------------------------------

        if self.worker_thread.is_alive():

            self.worker_thread.join(
                timeout=2.0
            )

        # --------------------------------------------------------------
        # OpenCV GUI 종료
        # --------------------------------------------------------------

        if self.show_window:

            cv2.destroyAllWindows()

            # GUI Event Queue에 남아 있는 종료 Event를
            # 한 번 처리합니다.
            try:
                cv2.waitKey(1)
            except cv2.error:
                pass

        return super().destroy_node()

    def publish_detections(
        self,
        result,
        frame_header,
    ) -> None:
        """
        Ultralytics YOLO 결과를
        vision_msgs/Detection2DArray 형태로 변환해 publish합니다.

        각 객체별로 전달되는 정보:
            - Bounding Box
            - 중심 Pixel (u, v)
            - Bounding Box 크기
            - Class 이름
            - Confidence

        frame_header에는 원본 RGB 영상의 timestamp가 들어 있으므로
        depth_position_node가 같은 시점의 Depth 프레임을 찾을 수 있습니다.
        """

        if frame_header is None:
            return

        detection_array = Detection2DArray()

        # 원본 RGB 영상의 timestamp와 frame_id를 그대로 사용합니다.
        detection_array.header = copy.deepcopy(
            frame_header
        )

        boxes = result.boxes

        # 검출된 객체가 없는 경우에도
        # 빈 Detection2DArray를 publish할 수 있습니다.
        if boxes is None:
            self.detection_publisher.publish(
                detection_array
            )
            return

        for box in boxes:

            # ----------------------------------------------------------
            # YOLO Bounding Box
            # ----------------------------------------------------------

            x1, y1, x2, y2 = (
                box.xyxy[0]
                .detach()
                .cpu()
                .tolist()
            )

            confidence = float(
                box.conf[0]
                .detach()
                .cpu()
            )

            class_id = int(
                box.cls[0]
                .detach()
                .cpu()
            )

            class_name = result.names.get(
                class_id,
                str(class_id),
            )

            # ----------------------------------------------------------
            # Bounding Box 중심
            # ----------------------------------------------------------

            center_u = (
                x1 + x2
            ) / 2.0

            center_v = (
                y1 + y2
            ) / 2.0

            width = (
                x2 - x1
            )

            height = (
                y2 - y1
            )

            # ----------------------------------------------------------
            # Detection2D 생성
            # ----------------------------------------------------------

            detection = Detection2D()

            detection.header = copy.deepcopy(
                frame_header
            )

            # BoundingBox2D의 center는 Pixel 좌표입니다.
            detection.bbox.center.position.x = (
                float(center_u)
            )

            detection.bbox.center.position.y = (
                float(center_v)
            )

            detection.bbox.center.theta = 0.0

            detection.bbox.size_x = float(
                width
            )

            detection.bbox.size_y = float(
                height
            )

            # ----------------------------------------------------------
            # Class / Confidence
            # ----------------------------------------------------------

            hypothesis = ObjectHypothesisWithPose()

            # 문자열 class 이름을 저장합니다.
            hypothesis.hypothesis.class_id = (
                str(class_name)
            )

            hypothesis.hypothesis.score = (
                confidence
            )

            detection.results.append(
                hypothesis
            )

            detection_array.detections.append(
                detection
            )

        # --------------------------------------------------------------
        # 최종 Detection Topic 발행
        # --------------------------------------------------------------

        self.detection_publisher.publish(
            detection_array
        )


# ======================================================================
# Main
# ======================================================================

def main(args=None) -> None:
    """
    실제 Python Main Thread입니다.

    기존 방식:

        rclpy.spin(node)

    에서는 Main Thread의 제어권이 ROS Executor에 계속 머뭅니다.

    이 코드에서는:

        spin_once()
            ↓
        display_once()
            ↓
        spin_once()
            ↓
        display_once()

    형태로 실행합니다.

    따라서 OpenCV HighGUI가 실제 Python Main Thread에서
    지속적으로 처리됩니다.
    """

    rclpy.init(
        args=args
    )

    node = None

    try:

        node = YoloDetectionNode()

        # ==============================================================
        # Main Event Loop
        # ==============================================================

        while (
            rclpy.ok()
            and node.running
        ):

            # ----------------------------------------------------------
            # ROS Callback 처리
            # ----------------------------------------------------------
            #
            # 최대 10 ms 동안 ROS Event를 기다립니다.
            #
            # 처리 가능한 Callback 예:
            #
            # - image_callback()
            # - report_statistics()
            #
            # Callback 처리가 끝나면 반드시 Main Thread로
            # 다시 제어권이 돌아옵니다.
            rclpy.spin_once(
                node,
                timeout_sec=0.01
            )

            # ----------------------------------------------------------
            # OpenCV GUI 처리
            # ----------------------------------------------------------
            #
            # Worker Thread가 아니라 실제 Main Thread에서
            # 실행되는 것이 중요합니다.
            node.display_once()

    except KeyboardInterrupt:

        # Ctrl+C 종료 시 ROS Context 상태에 따라
        # rosout logger 호출이 실패할 수 있으므로
        # 단순 출력으로 종료 사실을 알립니다.
        print(
            '\n사용자 요청으로 종료합니다.'
        )

    except Exception as error:

        if (
            node is not None
            and rclpy.ok()
        ):

            node.get_logger().fatal(
                f'노드 실행 실패: {error}'
            )

        else:

            print(
                f'노드 실행 실패: {error}'
            )

    finally:

        # --------------------------------------------------------------
        # Node 정리
        # --------------------------------------------------------------

        if node is not None:

            node.destroy_node()

        # --------------------------------------------------------------
        # ROS Context 정리
        # --------------------------------------------------------------

        if rclpy.ok():

            rclpy.shutdown()


if __name__ == '__main__':

    main()