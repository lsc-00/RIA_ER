#!/usr/bin/env python3

from pathlib import Path
import shutil
import subprocess
import sys


WORKSPACE = Path.home() / "ria/RIA_food_ws"

VIDEO_DIR = (
    WORKSPACE
    / "datasets"
    / "fried_chicken_stage_raw"
    / "videos"
)

OUTPUT_ROOT = (
    WORKSPACE
    / "datasets"
    / "fried_chicken_stage_work"
    / "extracted_frames"
)

# 기본 추출 간격입니다.
# 3.0이면 3초마다 한 장을 추출합니다.
INTERVAL_SECONDS = 0.8

SUPPORTED_EXTENSIONS = {
    ".mp4",
    ".mov",
    ".mkv",
    ".webm",
    ".avi",
    ".m4v",
}


def check_ffmpeg() -> None:
    """FFmpeg와 FFprobe 설치 여부를 확인합니다."""
    required_commands = ("ffmpeg", "ffprobe")

    for command in required_commands:
        if shutil.which(command) is None:
            raise RuntimeError(
                f"{command} 명령을 찾을 수 없습니다.\n"
                "다음 명령으로 설치하십시오:\n"
                "sudo apt update && sudo apt install -y ffmpeg"
            )


def get_video_duration(video_path: Path) -> float:
    """FFprobe를 이용해 영상 길이를 초 단위로 확인합니다."""
    command = [
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        str(video_path),
    ]

    result = subprocess.run(
        command,
        check=True,
        capture_output=True,
        text=True,
    )

    return float(result.stdout.strip())


def extract_frames(video_path: Path) -> int:
    """영상 하나에서 일정 시간 간격으로 프레임을 추출합니다."""
    output_dir = OUTPUT_ROOT / video_path.stem
    output_dir.mkdir(parents=True, exist_ok=True)

    # 이전 실행 결과와 섞이지 않게 기존 jpg 파일을 제거합니다.
    for old_image in output_dir.glob("*.jpg"):
        old_image.unlink()

    output_pattern = output_dir / f"{video_path.stem}_%06d.jpg"

    fps_filter = f"fps=1/{INTERVAL_SECONDS}"

    command = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "warning",
        "-i",
        str(video_path),
        "-vf",
        fps_filter,
        "-q:v",
        "2",
        "-start_number",
        "0",
        str(output_pattern),
    ]

    print("\n" + "=" * 70)
    print(f"[영상] {video_path.name}")

    try:
        duration = get_video_duration(video_path)
        expected_count = int(duration / INTERVAL_SECONDS) + 1

        print(f"[길이] {duration:.1f}초")
        print(f"[간격] {INTERVAL_SECONDS:.1f}초")
        print(f"[예상] 약 {expected_count}장")
        print(f"[출력] {output_dir}")

    except (subprocess.CalledProcessError, ValueError) as error:
        print(f"[경고] 영상 길이를 확인하지 못했습니다: {error}")

    subprocess.run(command, check=True)

    saved_count = len(list(output_dir.glob("*.jpg")))

    print(f"[완료] {saved_count}장 저장")

    return saved_count


def main() -> None:
    check_ffmpeg()

    if not VIDEO_DIR.exists():
        raise FileNotFoundError(
            f"영상 폴더가 없습니다:\n{VIDEO_DIR}"
        )

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    video_paths = sorted(
        path
        for path in VIDEO_DIR.iterdir()
        if path.is_file()
        and path.suffix.lower() in SUPPORTED_EXTENSIONS
    )

    if not video_paths:
        raise FileNotFoundError(
            f"지원되는 영상 파일이 없습니다:\n{VIDEO_DIR}"
        )

    print(f"검색된 영상: {len(video_paths)}개")

    total_count = 0
    failed_videos = []

    for video_path in video_paths:
        try:
            total_count += extract_frames(video_path)

        except subprocess.CalledProcessError as error:
            failed_videos.append(video_path.name)
            print(f"[실패] {video_path.name}: {error}")

    print("\n" + "=" * 70)
    print(f"처리한 영상: {len(video_paths)}개")
    print(f"추출된 전체 이미지: {total_count}장")

    if failed_videos:
        print("추출 실패 영상:")
        for name in failed_videos:
            print(f"  - {name}")
        sys.exit(1)

    print("모든 영상의 프레임 추출이 완료되었습니다.")


if __name__ == "__main__":
    main()