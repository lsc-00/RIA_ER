#!/usr/bin/env python3
"""
RealSense RGB-D 기반 batter/done 실시간 객체 검출 프로그램입니다.

주요 기능
1. RealSense의 Color 영상과 Depth 영상을 동시에 수신합니다.
2. YOLO 모델을 사용하여 batter와 done 객체를 검출합니다.
3. 각 객체의 바운딩박스 중심 픽셀 좌표 (u, v)를 계산합니다.
4. 해당 픽셀의 Depth 정보를 이용하여 카메라 좌표계 (X, Y, Z)를 계산합니다.
5. 검출 결과와 좌표를 영상 및 터미널에 표시합니다.

클래스 구성
    0: batter
    1: done

실행
    python3 ~/ria/RIA_food_ws/scripts/camera_detection_xyz.py

종료
    q
    ESC
    Ctrl+C
"""

from pathlib import Path
import signal
import sys
import time

import cv2
import numpy as np
import pyrealsense2 as rs
from ultralytics import YOLO


# ============================================================
# 1. 프로젝트 및 모델 경로 설정
# ============================================================

# 프로젝트의 최상위 작업 공간 경로입니다.
# Path.home()을 사용하므로 사용자 홈 디렉터리가 자동으로 적용됩니다.
WORKSPACE = Path.home() / "ria" / "RIA_food_ws"

# 학습 완료된 YOLO 모델 경로입니다.
# 해당 모델 내부에는 다음 클래스 정보가 저장되어 있어야 합니다.
#   0: batter
#   1: done
MODEL_PATH = (
    WORKSPACE
    / "models"
    / "batter_done_best.pt"
)


# ============================================================
# 2. RealSense 카메라 스트림 설정
# ============================================================

# Color 영상과 Depth 영상의 해상도입니다.
# 두 스트림을 동일한 해상도로 설정하면 정렬 과정이 단순해집니다.
WIDTH = 640
HEIGHT = 480

# 카메라 프레임 속도입니다.
# D435i에서 일반적으로 안정적으로 사용할 수 있는 값입니다.
FPS = 30


# ============================================================
# 3. YOLO 추론 설정
# ============================================================

# YOLO 모델에 입력되는 영상 크기입니다.
# 학습할 때 640 크기를 사용했으므로 추론에서도 동일한 값을 사용합니다.
IMAGE_SIZE = 640

# 이 값보다 낮은 신뢰도의 검출 결과는 무시합니다.
CONFIDENCE_THRESHOLD = 0.50

# 0은 첫 번째 CUDA GPU를 의미합니다.
# CPU를 사용하려면 "cpu"로 변경할 수 있습니다.
DEVICE = 0


# ============================================================
# 4. Depth 처리 설정
# ============================================================

# 바운딩박스 중심 픽셀 한 점만 사용하면,
# 반사나 깊이 결측으로 Depth가 0이 될 수 있습니다.
#
# 따라서 중심점 주변 영역의 Depth를 읽어 중앙값을 사용합니다.
#
# DEPTH_RADIUS = 4인 경우:
#   가로 9픽셀 × 세로 9픽셀 영역을 검사합니다.
DEPTH_RADIUS = 4

# 너무 가깝거나 너무 먼 Depth 값은 잘못된 값일 가능성이 있으므로 제외합니다.
MIN_DEPTH_M = 0.10
MAX_DEPTH_M = 3.00


# ============================================================
# 5. 화면 및 터미널 출력 설정
# ============================================================

# 터미널 출력이 너무 빠르게 반복되지 않도록 출력 간격을 제한합니다.
PRINT_INTERVAL_SECONDS = 0.2

# OpenCV 영상 창 이름입니다.
WINDOW_NAME = "Batter Done RGB-D Detection"


# ============================================================
# 6. 프로그램 실행 상태 변수
# ============================================================

# Ctrl+C, q, ESC 등의 종료 요청이 들어오면 False로 변경됩니다.
running = True


# ============================================================
# 7. 종료 신호 처리
# ============================================================

def handle_exit_signal(signum, frame):
    """
    Ctrl+C 또는 시스템 종료 신호를 처리합니다.

    프로그램을 즉시 강제 종료하지 않고 running 값을 False로 바꾸어
    RealSense 파이프라인과 OpenCV 창을 정상적으로 정리하도록 합니다.
    """

    del frame

    global running

    print(
        f"\n종료 신호 수신: signal={signum}",
        flush=True,
    )

    running = False


# ============================================================
# 8. RealSense 카메라 검색
# ============================================================

def find_camera_serial(context):
    """
    연결된 RealSense 카메라의 시리얼 번호를 검색합니다.

    처리 순서
    1. 연결된 모든 RealSense 장치를 확인합니다.
    2. D435i가 있으면 해당 장치를 우선 선택합니다.
    3. D435i가 없으면 첫 번째 RealSense 장치를 사용합니다.

    반환값
        선택된 카메라의 시리얼 번호
    """

    first_serial = None

    for device in context.query_devices():
        name = device.get_info(
            rs.camera_info.name
        )

        serial = device.get_info(
            rs.camera_info.serial_number
        )

        print(
            f"검색된 카메라: {name}, "
            f"시리얼: {serial}"
        )

        # D435i가 없을 경우를 대비해 첫 번째 장치의 시리얼을 저장합니다.
        if first_serial is None:
            first_serial = serial

        normalized_name = name.upper()

        # D435i 또는 D435iF 이름이 포함된 장치를 우선 선택합니다.
        if (
            "D435I" in normalized_name
            or "435I" in normalized_name
        ):
            print(
                f"D435i 카메라 선택: {serial}"
            )

            return serial

    # 연결된 RealSense 장치가 하나도 없으면 프로그램을 중단합니다.
    if first_serial is None:
        raise RuntimeError(
            "연결된 RealSense 카메라가 없습니다."
        )

    print(
        "D435i를 찾지 못해 "
        f"첫 번째 RealSense를 선택합니다: {first_serial}"
    )

    return first_serial


# ============================================================
# 9. RealSense 스트림 시작
# ============================================================

def start_camera():
    """
    RealSense Color 및 Depth 스트림을 시작합니다.

    중요한 점
    YOLO는 Color 영상에서 객체를 검출합니다.
    따라서 Depth 영상도 Color 영상 좌표계에 맞춰 정렬해야
    동일한 (u, v) 픽셀에서 올바른 Depth를 읽을 수 있습니다.

    반환값
        pipeline
            RealSense 프레임 수신 객체

        align
            Depth를 Color 기준으로 정렬하는 객체
    """

    context = rs.context()

    serial = find_camera_serial(context)

    pipeline = rs.pipeline()
    config = rs.config()

    # 여러 RealSense가 연결되어 있을 때 특정 장치를 선택합니다.
    config.enable_device(serial)

    # YOLO가 사용할 BGR Color 영상 스트림입니다.
    config.enable_stream(
        rs.stream.color,
        WIDTH,
        HEIGHT,
        rs.format.bgr8,
        FPS,
    )

    # 객체 거리 측정에 사용할 Depth 스트림입니다.
    config.enable_stream(
        rs.stream.depth,
        WIDTH,
        HEIGHT,
        rs.format.z16,
        FPS,
    )

    # 설정한 스트림으로 카메라를 시작합니다.
    pipeline.start(config)

    # YOLO 바운딩박스 좌표는 Color 영상 기준입니다.
    # 따라서 Depth 프레임을 Color 프레임에 맞춰 정렬합니다.
    align = rs.align(
        rs.stream.color
    )

    print("RealSense 스트림 시작 완료")

    return pipeline, align


# ============================================================
# 10. 안정적인 Depth 계산
# ============================================================

def robust_depth(
    depth_frame,
    u,
    v,
    radius=DEPTH_RADIUS,
):
    """
    중심 픽셀 주변의 유효 Depth 중앙값을 반환합니다.

    중심 픽셀 한 점의 Depth만 사용할 경우 발생 가능한 문제
    1. 반사로 인해 Depth가 0으로 측정될 수 있습니다.
    2. 객체 표면이 불규칙해 거리값이 튈 수 있습니다.
    3. 중심점이 객체 사이의 빈 공간에 위치할 수 있습니다.

    이를 보완하기 위해 중심 주변 영역의 유효한 Depth 값만 수집한 뒤
    평균보다 이상치에 강한 중앙값을 사용합니다.

    반환값
        정상 Depth가 존재할 경우:
            meter 단위의 float 값

        유효 Depth가 없을 경우:
            None
    """

    width = depth_frame.get_width()
    height = depth_frame.get_height()

    valid_depth_values = []

    # 검사할 ROI의 시작점과 끝점을 영상 범위 안으로 제한합니다.
    x_start = max(
        0,
        u - radius,
    )

    x_end = min(
        width,
        u + radius + 1,
    )

    y_start = max(
        0,
        v - radius,
    )

    y_end = min(
        height,
        v + radius + 1,
    )

    # 중심 주변의 모든 픽셀 Depth를 확인합니다.
    for pixel_v in range(
        y_start,
        y_end,
    ):
        for pixel_u in range(
            x_start,
            x_end,
        ):
            depth_m = depth_frame.get_distance(
                pixel_u,
                pixel_v,
            )

            # 설정한 유효 거리 범위 안에 있는 값만 사용합니다.
            if (
                MIN_DEPTH_M
                <= depth_m
                <= MAX_DEPTH_M
            ):
                valid_depth_values.append(
                    depth_m
                )

    # 주변 모든 픽셀에서 유효한 Depth를 얻지 못한 경우입니다.
    if not valid_depth_values:
        return None

    # 이상치 영향을 줄이기 위해 평균이 아닌 중앙값을 사용합니다.
    return float(
        np.median(
            valid_depth_values
        )
    )


# ============================================================
# 11. 픽셀 좌표를 카메라 3차원 좌표로 변환
# ============================================================

def deproject(
    intrinsics,
    u,
    v,
    depth_m,
):
    """
    2차원 픽셀 좌표 (u, v)와 Depth를
    카메라 좌표계 (X, Y, Z)로 변환합니다.

    입력값
        intrinsics
            카메라 내부 파라미터

        u, v
            Color 영상의 픽셀 좌표

        depth_m
            해당 픽셀까지의 거리[m]

    RealSense optical frame 좌표 방향
        +X: 영상 오른쪽
        +Y: 영상 아래쪽
        +Z: 카메라 전방

    주의
        이 좌표는 로봇 Base 좌표가 아니라 카메라 좌표입니다.
        로봇 좌표로 사용하려면 Hand-eye Calibration 변환이 필요합니다.
    """

    point = rs.rs2_deproject_pixel_to_point(
        intrinsics,
        [
            float(u),
            float(v),
        ],
        float(depth_m),
    )

    return (
        float(point[0]),
        float(point[1]),
        float(point[2]),
    )


# ============================================================
# 12. 클래스별 표시 색상 지정
# ============================================================

def class_color(class_id):
    """
    클래스에 따라 바운딩박스 표시 색상을 반환합니다.

    OpenCV 색상 순서는 RGB가 아니라 BGR입니다.
    """

    if class_id == 0:
        # batter: 노란색
        return 0, 255, 255

    if class_id == 1:
        # done: 초록색
        return 0, 255, 0

    # 알 수 없는 클래스: 흰색
    return 255, 255, 255


# ============================================================
# 13. 검출 결과 화면 표시
# ============================================================

def draw_detections(
    image,
    depth_frame,
    intrinsics,
    result,
):
    """
    YOLO 검출 결과를 영상에 표시합니다.

    객체별 처리 순서
    1. 바운딩박스 좌표를 읽습니다.
    2. 클래스 번호와 신뢰도를 읽습니다.
    3. 바운딩박스 중심 픽셀 (u, v)를 계산합니다.
    4. 중심 주변 Depth 중앙값을 구합니다.
    5. 픽셀 좌표를 카메라 좌표 (X, Y, Z)로 변환합니다.
    6. 화면과 터미널에 결과를 표시합니다.

    반환값
        display
            검출 결과가 그려진 영상

        messages
            터미널에 출력할 객체 정보 문자열 목록
    """

    display = image.copy()
    messages = []

    boxes = result.boxes

    # 검출된 객체가 없는 경우 화면에 메시지만 표시합니다.
    if boxes is None or len(boxes) == 0:
        cv2.putText(
            display,
            "No detection",
            (15, 65),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (0, 0, 255),
            2,
        )

        return display, messages

    image_height, image_width = (
        display.shape[:2]
    )

    # 검출된 객체를 하나씩 처리합니다.
    for object_index, box in enumerate(
        boxes,
        start=1,
    ):
        # 바운딩박스 좌표를 가져옵니다.
        x1, y1, x2, y2 = (
            box.xyxy[0]
            .cpu()
            .tolist()
        )

        # YOLO가 예측한 객체 신뢰도입니다.
        confidence = float(
            box.conf[0].cpu()
        )

        # YOLO가 예측한 클래스 번호입니다.
        class_id = int(
            box.cls[0].cpu()
        )

        # 클래스 이름은 코드에 고정하지 않고
        # 학습된 PT 모델 내부의 names 정보를 사용합니다.
        class_name = result.names.get(
            class_id,
            f"class_{class_id}",
        )

        # OpenCV 그리기에 사용하기 위해 실수 좌표를 정수로 변환합니다.
        # 영상 범위를 벗어나지 않도록 값도 제한합니다.
        x1_i = max(
            0,
            min(
                image_width - 1,
                int(round(x1)),
            ),
        )

        y1_i = max(
            0,
            min(
                image_height - 1,
                int(round(y1)),
            ),
        )

        x2_i = max(
            0,
            min(
                image_width - 1,
                int(round(x2)),
            ),
        )

        y2_i = max(
            0,
            min(
                image_height - 1,
                int(round(y2)),
            ),
        )

        # 바운딩박스 중심 픽셀 좌표를 계산합니다.
        u = max(
            0,
            min(
                image_width - 1,
                int(
                    round(
                        (x1 + x2) / 2
                    )
                ),
            ),
        )

        v = max(
            0,
            min(
                image_height - 1,
                int(
                    round(
                        (y1 + y2) / 2
                    )
                ),
            ),
        )

        # 중심점 주변에서 안정적인 Depth 값을 계산합니다.
        depth_m = robust_depth(
            depth_frame,
            u,
            v,
        )

        # 유효한 Depth가 있을 때만 카메라 좌표를 계산합니다.
        if depth_m is not None:
            xyz = deproject(
                intrinsics,
                u,
                v,
                depth_m,
            )
        else:
            xyz = None

        color = class_color(
            class_id
        )

        # 객체의 바운딩박스를 그립니다.
        cv2.rectangle(
            display,
            (x1_i, y1_i),
            (x2_i, y2_i),
            color,
            2,
        )

        # 바운딩박스 중심점을 빨간색 원으로 표시합니다.
        cv2.circle(
            display,
            (u, v),
            6,
            (0, 0, 255),
            -1,
        )

        # 첫 번째 줄에는 클래스, 신뢰도, 픽셀 좌표를 표시합니다.
        line_1 = (
            f"{class_name} "
            f"{confidence:.2f} "
            f"uv=({u},{v})"
        )

        if xyz is None:
            # Depth를 얻지 못했을 경우의 표시입니다.
            line_2 = (
                "XYZ=(depth unavailable)"
            )

            message = (
                f"객체 {object_index}: "
                f"class={class_name}, "
                f"id={class_id}, "
                f"conf={confidence:.3f}, "
                f"uv=({u},{v}), "
                "XYZ=unavailable"
            )

        else:
            x_m, y_m, z_m = xyz

            # 두 번째 줄에는 카메라 기준 XYZ 좌표를 표시합니다.
            line_2 = (
                f"XYZ=("
                f"{x_m:+.3f}, "
                f"{y_m:+.3f}, "
                f"{z_m:.3f}"
                f") m"
            )

            message = (
                f"객체 {object_index}: "
                f"class={class_name}, "
                f"id={class_id}, "
                f"conf={confidence:.3f}, "
                f"uv=({u},{v}), "
                f"XYZ=("
                f"{x_m:+.3f},"
                f"{y_m:+.3f},"
                f"{z_m:.3f}"
                f") m"
            )

        # 바운딩박스 위쪽에 문구를 배치합니다.
        # 영상 윗부분을 벗어나지 않도록 최소 y 위치를 지정합니다.
        text_y_1 = max(
            22,
            y1_i - 28,
        )

        text_y_2 = max(
            44,
            y1_i - 6,
        )

        cv2.putText(
            display,
            line_1,
            (x1_i, text_y_1),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52,
            color,
            2,
            cv2.LINE_AA,
        )

        cv2.putText(
            display,
            line_2,
            (x1_i, text_y_2),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52,
            color,
            2,
            cv2.LINE_AA,
        )

        messages.append(
            message
        )

    return display, messages


# ============================================================
# 14. 메인 실행 함수
# ============================================================

def main():
    """
    프로그램 전체 실행 흐름을 관리합니다.

    처리 순서
    1. 종료 신호를 등록합니다.
    2. YOLO 모델 파일을 확인하고 불러옵니다.
    3. RealSense Color/Depth 스트림을 시작합니다.
    4. 매 프레임마다 YOLO 추론을 실행합니다.
    5. 객체별 (u, v), (X, Y, Z)를 계산합니다.
    6. 화면과 터미널에 결과를 출력합니다.
    7. 종료 시 카메라와 창을 정리합니다.
    """

    global running

    pipeline = None

    # Ctrl+C 및 시스템 종료 요청을 처리합니다.
    signal.signal(
        signal.SIGINT,
        handle_exit_signal,
    )

    signal.signal(
        signal.SIGTERM,
        handle_exit_signal,
    )

    # 지정한 모델 파일이 존재하지 않으면 추론을 시작할 수 없습니다.
    if not MODEL_PATH.exists():
        print(
            f"모델 파일이 없습니다: {MODEL_PATH}",
            file=sys.stderr,
        )

        return 1

    try:
        # 학습된 YOLO 모델을 불러옵니다.
        print(
            f"YOLO 모델 로딩: {MODEL_PATH}"
        )

        model = YOLO(
            str(MODEL_PATH)
        )

        # 모델 내부에 저장된 클래스 구성을 출력합니다.
        # 정상이라면 {0: 'batter', 1: 'done'} 형태여야 합니다.
        print(
            f"모델 클래스: {model.names}"
        )

        # RealSense Color/Depth 스트림을 시작합니다.
        pipeline, align = start_camera()

        # OpenCV 출력 창을 생성합니다.
        cv2.namedWindow(
            WINDOW_NAME,
            cv2.WINDOW_NORMAL,
        )

        cv2.resizeWindow(
            WINDOW_NAME,
            960,
            720,
        )

        previous_time = (
            time.perf_counter()
        )

        last_print_time = 0.0

        # 사용자가 종료를 요청할 때까지 반복합니다.
        while running:
            # Color와 Depth 프레임이 모두 들어올 때까지 기다립니다.
            frames = pipeline.wait_for_frames(
                timeout_ms=5000
            )

            # Depth 프레임을 Color 영상 좌표계에 맞게 정렬합니다.
            aligned_frames = align.process(
                frames
            )

            depth_frame = (
                aligned_frames
                .get_depth_frame()
            )

            color_frame = (
                aligned_frames
                .get_color_frame()
            )

            # 두 프레임 중 하나라도 없으면 해당 반복을 건너뜁니다.
            if (
                not depth_frame
                or not color_frame
            ):
                continue

            # RealSense Color 프레임을 OpenCV에서 사용할 NumPy 배열로 변환합니다.
            image = np.asanyarray(
                color_frame.get_data()
            )

            # 픽셀 좌표를 카메라 3차원 좌표로 변환할 때 필요한
            # Color 카메라 내부 파라미터입니다.
            intrinsics = (
                color_frame
                .profile
                .as_video_stream_profile()
                .intrinsics
            )

            # 현재 Color 영상에 대해 YOLO 추론을 수행합니다.
            result = model.predict(
                source=image,
                imgsz=IMAGE_SIZE,
                conf=CONFIDENCE_THRESHOLD,
                device=DEVICE,
                verbose=False,
            )[0]

            # 검출 결과와 XYZ 좌표를 영상 위에 표시합니다.
            display, messages = draw_detections(
                image,
                depth_frame,
                intrinsics,
                result,
            )

            # 실제 처리 속도를 계산합니다.
            current_time = (
                time.perf_counter()
            )

            elapsed_time = max(
                current_time
                - previous_time,
                1e-9,
            )

            measured_fps = (
                1.0 / elapsed_time
            )

            previous_time = current_time

            # 화면 왼쪽 상단에 처리 FPS를 표시합니다.
            cv2.putText(
                display,
                f"FPS: {measured_fps:.1f}",
                (15, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.75,
                (255, 255, 255),
                2,
            )

            # 화면 아래에 카메라 좌표축 방향을 표시합니다.
            cv2.putText(
                display,
                (
                    "Camera frame: "
                    "+X right, "
                    "+Y down, "
                    "+Z forward"
                ),
                (
                    15,
                    display.shape[0] - 18,
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.48,
                (255, 255, 255),
                2,
            )

            # 최종 영상을 화면에 표시합니다.
            cv2.imshow(
                WINDOW_NAME,
                display,
            )

            # 객체 정보가 너무 빠르게 출력되지 않도록 일정 간격으로 출력합니다.
            if (
                messages
                and current_time
                - last_print_time
                >= PRINT_INTERVAL_SECONDS
            ):
                print(
                    "\r"
                    + " | ".join(messages)
                    + " " * 10,
                    end="",
                    flush=True,
                )

                last_print_time = (
                    current_time
                )

            # q 또는 ESC 입력을 확인합니다.
            key = cv2.waitKey(1) & 0xFF

            if key in (
                ord("q"),
                27,
            ):
                running = False

            # 사용자가 OpenCV 창의 X 버튼을 눌렀는지 확인합니다.
            try:
                window_visible = (
                    cv2.getWindowProperty(
                        WINDOW_NAME,
                        cv2.WND_PROP_VISIBLE,
                    )
                )

                if window_visible < 1:
                    running = False

            except cv2.error:
                running = False

        return 0

    except Exception as error:
        print(
            f"\n오류: {error}",
            file=sys.stderr,
        )

        return 1

    finally:
        # 프로그램이 종료될 때 RealSense 파이프라인을 반드시 정리합니다.
        if pipeline is not None:
            try:
                pipeline.stop()

            except RuntimeError:
                pass

        # 열려 있는 OpenCV 창을 모두 닫습니다.
        cv2.destroyAllWindows()

        print(
            "\n프로그램이 종료되었습니다."
        )


if __name__ == "__main__":
    raise SystemExit(
        main()
    )