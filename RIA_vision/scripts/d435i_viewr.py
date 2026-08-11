# #!/usr/bin/env python3

# import signal
# import sys

# import cv2
# import numpy as np
# import pyrealsense2 as rs


# running = True


# def signal_handler(signum, frame):
#     """Ctrl+C 또는 종료 신호를 받으면 반복문을 종료합니다."""
#     global running
#     print("\n종료 신호를 받았습니다.")
#     running = False


# def find_d435i(context):
#     """연결된 RealSense 장치 중 D435i의 시리얼 번호를 찾습니다."""
#     for device in context.query_devices():
#         try:
#             name = device.get_info(rs.camera_info.name)
#             serial = device.get_info(rs.camera_info.serial_number)

#             print(f"검색된 카메라: {name}, 시리얼: {serial}")

#             normalized_name = name.upper()

#             if "D435I" in normalized_name or "435I" in normalized_name:
#                 return serial

#         except RuntimeError as error:
#             print(f"장치 정보 확인 실패: {error}")

#     return None


# def main():
#     global running

#     signal.signal(signal.SIGINT, signal_handler)
#     signal.signal(signal.SIGTERM, signal_handler)

#     pipeline = rs.pipeline()
#     config = rs.config()
#     context = rs.context()

#     pipeline_started = False

#     try:
#         d435i_serial = find_d435i(context)

#         if d435i_serial is None:
#             raise RuntimeError("Intel RealSense D435i를 찾지 못했습니다.")

#         print(f"D435i 선택 완료: {d435i_serial}")

#         # 여러 RealSense 카메라가 연결된 경우 특정 D435i만 선택합니다.
#         config.enable_device(d435i_serial)

#         # D435i Depth 스트림 설정
#         config.enable_stream(
#             rs.stream.depth,
#             640,
#             480,
#             rs.format.z16,
#             30,
#         )

#         print("D435i Depth 스트림을 시작합니다.")
#         print("종료: q, Esc 또는 Ctrl+C")

#         profile = pipeline.start(config)
#         pipeline_started = True

#         device = profile.get_device()

#         try:
#             usb_type = device.get_info(
#                 rs.camera_info.usb_type_descriptor
#             )
#             print(f"USB 연결 형식: {usb_type}")

#             if usb_type.startswith("2"):
#                 print(
#                     "경고: USB 2.x로 연결되어 있습니다. "
#                     "프레임 수신이 불안정하면 해상도나 FPS를 낮추십시오."
#                 )

#         except RuntimeError:
#             print("USB 연결 형식을 확인할 수 없습니다.")

#         colorizer = rs.colorizer()

#         while running:
#             try:
#                 frames = pipeline.wait_for_frames(timeout_ms=5000)

#             except RuntimeError as error:
#                 print(f"프레임 수신 실패: {error}")
#                 print("USB 연결, 카메라 점유 상태 또는 스트림 설정을 확인하십시오.")
#                 break

#             depth_frame = frames.get_depth_frame()

#             if not depth_frame:
#                 print("Depth 프레임이 누락되었습니다.")
#                 continue

#             # 16비트 Depth 데이터를 화면 표시용 컬러맵으로 변환합니다.
#             depth_color_frame = colorizer.colorize(depth_frame)
#             depth_image = np.asanyarray(
#                 depth_color_frame.get_data()
#             )

#             # 화면 중앙 거리 표시
#             width = depth_frame.get_width()
#             height = depth_frame.get_height()

#             center_x = width // 2
#             center_y = height // 2

#             center_distance = depth_frame.get_distance(
#                 center_x,
#                 center_y,
#             )

#             cv2.circle(
#                 depth_image,
#                 (center_x, center_y),
#                 5,
#                 (255, 255, 255),
#                 -1,
#             )

#             cv2.putText(
#                 depth_image,
#                 f"Center distance: {center_distance:.3f} m",
#                 (10, 30),
#                 cv2.FONT_HERSHEY_SIMPLEX,
#                 0.7,
#                 (255, 255, 255),
#                 2,
#                 cv2.LINE_AA,
#             )

#             cv2.imshow(
#                 "Intel RealSense D435i Depth",
#                 depth_image,
#             )

#             key = cv2.waitKey(1) & 0xFF

#             if key in (ord("q"), 27):
#                 print("키 입력으로 종료합니다.")
#                 running = False

#     except Exception as error:
#         print(f"오류: {error}")
#         sys.exit(1)

#     finally:
#         if pipeline_started:
#             try:
#                 pipeline.stop()
#             except RuntimeError:
#                 pass

#         cv2.destroyAllWindows()

#         # OpenCV GUI 이벤트가 남지 않도록 한 번 처리합니다.
#         cv2.waitKey(1)

#         print("카메라와 화면을 종료했습니다.")


# if __name__ == "__main__":
#     main()

import signal
import sys

import cv2
import numpy as np
import pyrealsense2 as rs


running = True


def signal_handler(signum, frame):
    """Ctrl+C 또는 종료 신호를 받으면 반복문을 종료합니다."""
    global running
    print("\n종료 신호를 받았습니다.")
    running = False


def find_d435i(context):
    """연결된 RealSense 장치 중 D435i의 시리얼 번호를 찾습니다."""
    for device in context.query_devices():
        try:
            name = device.get_info(rs.camera_info.name)
            serial = device.get_info(rs.camera_info.serial_number)

            print(f"검색된 카메라: {name}, 시리얼: {serial}")

            normalized_name = name.upper()

            if "D435I" in normalized_name or "435I" in normalized_name:
                return serial

        except RuntimeError as error:
            print(f"장치 정보 확인 실패: {error}")

    return None


def main():
    global running

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    pipeline = rs.pipeline()
    config = rs.config()
    context = rs.context()

    pipeline_started = False

    try:
        d435i_serial = find_d435i(context)

        if d435i_serial is None:
            raise RuntimeError("Intel RealSense D435i를 찾지 못했습니다.")

        print(f"D435i 선택 완료: {d435i_serial}")

        config.enable_device(d435i_serial)

        # Depth 스트림
        config.enable_stream(
            rs.stream.depth,
            640,
            480,
            rs.format.z16,
            30,
        )

        # 일반 컬러 카메라 스트림
        config.enable_stream(
            rs.stream.color,
            640,
            480,
            rs.format.bgr8,
            30,
        )

        print("D435i Color + Depth 스트림을 시작합니다.")
        print("종료: q, Esc 또는 Ctrl+C")

        profile = pipeline.start(config)
        pipeline_started = True

        device = profile.get_device()

        try:
            usb_type = device.get_info(
                rs.camera_info.usb_type_descriptor
            )
            print(f"USB 연결 형식: {usb_type}")

        except RuntimeError:
            print("USB 연결 형식을 확인할 수 없습니다.")

        while running:
            try:
                frames = pipeline.wait_for_frames(timeout_ms=5000)

            except RuntimeError as error:
                print(f"프레임 수신 실패: {error}")
                break

            depth_frame = frames.get_depth_frame()
            color_frame = frames.get_color_frame()

            if not depth_frame:
                print("Depth 프레임이 누락되었습니다.")
                continue

            if not color_frame:
                print("Color 프레임이 누락되었습니다.")
                continue

            # 일반 카메라 영상
            color_image = np.asanyarray(
                color_frame.get_data()
            )

            # 화면 중앙 좌표
            width = depth_frame.get_width()
            height = depth_frame.get_height()

            center_x = width // 2
            center_y = height // 2

            # 중앙 거리 측정
            center_distance = depth_frame.get_distance(
                center_x,
                center_y,
            )

            cv2.circle(
                color_image,
                (center_x, center_y),
                5,
                (0, 0, 255),
                -1,
            )

            cv2.putText(
                color_image,
                f"Center distance: {center_distance:.3f} m",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )

            cv2.imshow(
                "Intel RealSense D435i Color",
                color_image,
            )

            key = cv2.waitKey(1) & 0xFF

            if key in (ord("q"), 27):
                print("키 입력으로 종료합니다.")
                running = False

    except Exception as error:
        print(f"오류: {error}")
        sys.exit(1)

    finally:
        if pipeline_started:
            try:
                pipeline.stop()
            except RuntimeError:
                pass

        cv2.destroyAllWindows()
        cv2.waitKey(1)

        print("카메라와 화면을 종료했습니다.")


if __name__ == "__main__":
    main()