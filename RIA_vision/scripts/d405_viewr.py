#!/usr/bin/env python3

import signal
import sys
import cv2
import numpy as np
import pyrealsense2 as rs


running = True


def signal_handler(signum, frame):
    global running
    print("\n종료 신호를 받았습니다.")
    running = False


def main():
    global running

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    pipeline = rs.pipeline()
    config = rs.config()
    context = rs.context()

    pipeline_started = False

    try:
        d405_serial = None

        for device in context.query_devices():
            name = device.get_info(rs.camera_info.name)
            serial = device.get_info(rs.camera_info.serial_number)

            print(f"검색된 카메라: {name}, 시리얼: {serial}")

            if "D405" in name or "405" in name:
                d405_serial = serial
                break

        if d405_serial is None:
            raise RuntimeError("RealSense D405를 찾지 못했습니다.")

        print(f"D405 선택 완료: {d405_serial}")

        config.enable_device(d405_serial)

        # USB 2.0에서도 비교적 안정적인 설정
        config.enable_stream(
            rs.stream.depth,
            480,
            270,
            rs.format.z16,
            30,
        )

        print("Depth 스트림을 시작합니다.")
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

        colorizer = rs.colorizer()

        while running:
            try:
                frames = pipeline.wait_for_frames(timeout_ms=5000)
            except RuntimeError as error:
                print(f"프레임 수신 실패: {error}")
                break

            depth_frame = frames.get_depth_frame()

            if not depth_frame:
                print("Depth 프레임이 누락되었습니다.")
                continue

            depth_color_frame = colorizer.colorize(depth_frame)
            depth_image = np.asanyarray(
                depth_color_frame.get_data()
            )

            cv2.imshow(
                "Intel RealSense D405 Depth",
                depth_image,
            )

            key = cv2.waitKey(1) & 0xFF

            if key == ord("q") or key == 27:
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