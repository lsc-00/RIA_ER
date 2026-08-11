#!/usr/bin/env python3
"""
RealSense RGB 카메라 영상에서 튀긴 치킨을 실시간 검출하고
바운딩박스 및 중심 픽셀 좌표 (u, v)를 표시합니다.

실행:
    python3 camera_detection.py

종료:
    q
    ESC
    Ctrl+C
"""

from pathlib import Path
import signal
import sys
import threading
import time

import cv2
from ultralytics import YOLO


# ============================================================
# 설정
# ============================================================

WORKSPACE = Path.home() / "ria" / "RIA_food_ws"
MODEL_PATH = WORKSPACE / "models" / "fried_chicken_best.pt"

# RealSense RGB 장치
CAMERA_DEVICE = "/dev/video4"

FRAME_WIDTH = 1280
FRAME_HEIGHT = 720
CAMERA_FPS = 30

# YOLO 설정
IMAGE_SIZE = 640
CONFIDENCE_THRESHOLD = 0.50
DEVICE = 0

WINDOW_NAME = "Fried Chicken Detection"

# 터미널 출력 빈도 제한
PRINT_INTERVAL_SECONDS = 0.2


# ============================================================
# 종료 제어
# ============================================================

stop_event = threading.Event()


def handle_exit_signal(signum, frame) -> None:
    """Ctrl+C 또는 종료 신호를 받으면 종료 플래그를 설정합니다."""
    del frame

    print(f"\n종료 신호 수신: signal={signum}", flush=True)
    stop_event.set()


# ============================================================
# 카메라 프레임 읽기 스레드
# ============================================================

class CameraReader:
    """
    cap.read()를 별도 스레드에서 수행합니다.

    카메라 드라이버 내부에서 read()가 지연되더라도 메인 스레드는
    Ctrl+C와 화면 입력을 계속 처리할 수 있습니다.
    """

    def __init__(self, capture: cv2.VideoCapture) -> None:
        self.capture = capture

        self.lock = threading.Lock()
        self.latest_frame = None
        self.frame_received = False
        self.read_failed = False

        self.thread = threading.Thread(
            target=self._read_loop,
            name="camera-reader",
            daemon=True,
        )

    def start(self) -> None:
        self.thread.start()

    def _read_loop(self) -> None:
        while not stop_event.is_set():
            success, frame = self.capture.read()

            if not success:
                self.read_failed = True
                time.sleep(0.05)
                continue

            with self.lock:
                self.latest_frame = frame
                self.frame_received = True
                self.read_failed = False

    def get_latest_frame(self):
        with self.lock:
            if self.latest_frame is None:
                return None

            return self.latest_frame.copy()


# ============================================================
# 카메라 열기
# ============================================================

def open_camera() -> cv2.VideoCapture:
    """V4L2 카메라를 열고 영상 형식과 해상도를 설정합니다."""

    print(f"카메라 연결 시도: {CAMERA_DEVICE}", flush=True)

    cap = cv2.VideoCapture(
        CAMERA_DEVICE,
        cv2.CAP_V4L2,
    )

    if not cap.isOpened():
        raise RuntimeError(
            f"{CAMERA_DEVICE} 카메라를 열 수 없습니다."
        )

    # ffplay에서 확인한 MJPEG 1280×720 30 FPS 설정
    cap.set(
        cv2.CAP_PROP_FOURCC,
        cv2.VideoWriter_fourcc(*"MJPG"),
    )
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
    cap.set(cv2.CAP_PROP_FPS, CAMERA_FPS)

    # 내부 버퍼를 줄여 오래된 프레임이 누적되는 현상 완화
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    actual_width = int(
        cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    )
    actual_height = int(
        cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    )
    actual_fps = cap.get(cv2.CAP_PROP_FPS)

    fourcc_value = int(
        cap.get(cv2.CAP_PROP_FOURCC)
    )

    actual_fourcc = "".join(
        chr((fourcc_value >> (8 * index)) & 0xFF)
        for index in range(4)
    )

    print("카메라가 정상적으로 열렸습니다.", flush=True)
    print(
        f"실제 해상도: "
        f"{actual_width} x {actual_height}",
        flush=True,
    )
    print(
        f"실제 FPS 설정값: {actual_fps:.1f}",
        flush=True,
    )
    print(
        f"실제 영상 포맷: {actual_fourcc}",
        flush=True,
    )

    return cap


# ============================================================
# 검출 결과 그리기
# ============================================================

def draw_detections(
    frame,
    result,
):
    """
    검출 결과를 영상에 표시하고 검출 정보를 반환합니다.

    반환값:
        display_frame
        detection_messages
    """

    display = frame.copy()
    detection_messages = []

    boxes = result.boxes

    if boxes is None or len(boxes) == 0:
        cv2.putText(
            display,
            "No detection",
            (15, 65),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (0, 0, 255),
            2,
            cv2.LINE_AA,
        )

        return display, detection_messages

    for object_index, box in enumerate(
        boxes,
        start=1,
    ):
        x1, y1, x2, y2 = (
            box.xyxy[0].cpu().tolist()
        )

        confidence = float(
            box.conf[0].cpu()
        )

        class_id = int(
            box.cls[0].cpu()
        )

        class_name = result.names[class_id]

        x1_i = int(round(x1))
        y1_i = int(round(y1))
        x2_i = int(round(x2))
        y2_i = int(round(y2))

        center_u = int(
            round((x1 + x2) / 2.0)
        )
        center_v = int(
            round((y1 + y2) / 2.0)
        )

        # 바운딩박스
        cv2.rectangle(
            display,
            (x1_i, y1_i),
            (x2_i, y2_i),
            (0, 255, 0),
            2,
        )

        # 중심점
        cv2.circle(
            display,
            (center_u, center_v),
            7,
            (0, 0, 255),
            -1,
        )

        # 중심점 십자선
        cv2.line(
            display,
            (center_u - 12, center_v),
            (center_u + 12, center_v),
            (0, 0, 255),
            2,
        )
        cv2.line(
            display,
            (center_u, center_v - 12),
            (center_u, center_v + 12),
            (0, 0, 255),
            2,
        )

        label = (
            f"{class_name} "
            f"{confidence:.2f} "
            f"center=({center_u},{center_v})"
        )

        text_y = max(30, y1_i - 10)

        cv2.putText(
            display,
            label,
            (x1_i, text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )

        message = (
            f"객체 {object_index}: "
            f"class={class_name}, "
            f"conf={confidence:.3f}, "
            f"center=({center_u}, {center_v}), "
            f"box=({x1_i}, {y1_i}, "
            f"{x2_i}, {y2_i})"
        )

        detection_messages.append(message)

    return display, detection_messages


# ============================================================
# 메인 함수
# ============================================================

def main() -> int:
    cap = None

    # Ctrl+C와 일반 종료 요청 처리
    signal.signal(
        signal.SIGINT,
        handle_exit_signal,
    )
    signal.signal(
        signal.SIGTERM,
        handle_exit_signal,
    )

    if not MODEL_PATH.exists():
        print(
            f"모델 파일이 없습니다: {MODEL_PATH}",
            file=sys.stderr,
        )
        return 1

    try:
        print(
            f"YOLO 모델 로딩: {MODEL_PATH}",
            flush=True,
        )

        model = YOLO(str(MODEL_PATH))

        print(
            "YOLO 모델 로딩 완료",
            flush=True,
        )

        cap = open_camera()

        camera_reader = CameraReader(cap)
        camera_reader.start()

        # OpenCV 창을 먼저 명시적으로 생성
        cv2.namedWindow(
            WINDOW_NAME,
            cv2.WINDOW_NORMAL,
        )

        cv2.resizeWindow(
            WINDOW_NAME,
            960,
            540,
        )

        print(
            "카메라 프레임 대기 중...",
            flush=True,
        )
        print(
            "종료: q, ESC 또는 Ctrl+C",
            flush=True,
        )

        previous_time = time.perf_counter()
        last_print_time = 0.0
        waiting_start_time = time.perf_counter()

        while not stop_event.is_set():
            frame = camera_reader.get_latest_frame()

            if frame is None:
                elapsed = (
                    time.perf_counter()
                    - waiting_start_time
                )

                if elapsed > 5.0:
                    print(
                        "\r카메라 프레임을 "
                        "5초 이상 기다리는 중입니다.",
                        end="",
                        flush=True,
                    )

                # 빈 창이라도 키보드 입력과 GUI 이벤트 처리
                key = cv2.waitKey(10) & 0xFF

                if key in (ord("q"), 27):
                    stop_event.set()

                time.sleep(0.01)
                continue

            # YOLO 실시간 추론
            result = model.predict(
                source=frame,
                imgsz=IMAGE_SIZE,
                conf=CONFIDENCE_THRESHOLD,
                device=DEVICE,
                verbose=False,
            )[0]

            display, detection_messages = (
                draw_detections(
                    frame,
                    result,
                )
            )

            current_time = time.perf_counter()

            elapsed = max(
                current_time - previous_time,
                1e-9,
            )

            measured_fps = 1.0 / elapsed
            previous_time = current_time

            cv2.putText(
                display,
                f"FPS: {measured_fps:.1f}",
                (15, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

            cv2.putText(
                display,
                "Press q or ESC to quit",
                (15, display.shape[0] - 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

            # 실제 카메라 화면 출력
            cv2.imshow(
                WINDOW_NAME,
                display,
            )

            # 터미널 출력이 너무 빠르게 반복되지 않도록 제한
            if (
                detection_messages
                and current_time - last_print_time
                >= PRINT_INTERVAL_SECONDS
            ):
                print(
                    "\r"
                    + " | ".join(detection_messages)
                    + " " * 10,
                    end="",
                    flush=True,
                )

                last_print_time = current_time

            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                print(
                    "\nq 키로 종료합니다.",
                    flush=True,
                )
                stop_event.set()

            elif key == 27:
                print(
                    "\nESC 키로 종료합니다.",
                    flush=True,
                )
                stop_event.set()

            # 사용자가 창의 X 버튼을 누른 경우
            try:
                window_visible = cv2.getWindowProperty(
                    WINDOW_NAME,
                    cv2.WND_PROP_VISIBLE,
                )

                if window_visible < 1:
                    print(
                        "\n카메라 창이 닫혔습니다.",
                        flush=True,
                    )
                    stop_event.set()

            except cv2.error:
                stop_event.set()

        return 0

    except KeyboardInterrupt:
        print(
            "\nKeyboardInterrupt로 종료합니다.",
            flush=True,
        )
        stop_event.set()
        return 130

    except Exception as error:
        print(
            f"\n오류: {error}",
            file=sys.stderr,
            flush=True,
        )
        stop_event.set()
        return 1

    finally:
        print(
            "\n카메라와 창을 정리합니다.",
            flush=True,
        )

        stop_event.set()

        if cap is not None:
            cap.release()

        cv2.destroyAllWindows()

        # OpenCV 창 종료 이벤트 처리 시간 확보
        for _ in range(5):
            cv2.waitKey(1)

        print(
            "프로그램이 종료되었습니다.",
            flush=True,
        )


if __name__ == "__main__":
    raise SystemExit(main())