#!/usr/bin/env python3
from pathlib import Path
import sys
import shutil

# ============================================================
# 사용자 설정
# ============================================================

DATASET_DIR = Path.home() / "ria" / "RIA_food_ws" / "datasets" / "fried_chicken_stage_raw" /"roboflow_done"

OLD_CLASS_ID = "0"
NEW_CLASS_ID = "1"

MAKE_BACKUP = True  # True면 .bak 백업 파일 생성


def find_label_files(dataset_dir: Path):
    """labels 폴더 안의 txt 파일들을 모두 찾습니다."""
    return sorted(
        path for path in dataset_dir.rglob("*.txt")
        if "labels" in path.parts
    )


def convert_class_id(label_file: Path):
    """하나의 라벨 파일에서 클래스 ID를 변경합니다."""
    changed = False
    output_lines = []

    lines = label_file.read_text(encoding="utf-8").splitlines()

    for line in lines:
        stripped = line.strip()

        # 빈 줄은 그대로 유지
        if not stripped:
            output_lines.append(line)
            continue

        parts = stripped.split()

        # YOLO Detection 형식은 최소 5개 값
        if len(parts) < 5:
            output_lines.append(line)
            continue

        if parts[0] == OLD_CLASS_ID:
            parts[0] = NEW_CLASS_ID
            changed = True

        output_lines.append(" ".join(parts))

    if changed:
        if MAKE_BACKUP:
            backup_path = label_file.with_suffix(label_file.suffix + ".bak")
            if not backup_path.exists():
                shutil.copy2(label_file, backup_path)

        label_file.write_text("\n".join(output_lines) + "\n", encoding="utf-8")

    return changed


def main():
    if not DATASET_DIR.exists():
        print(f"데이터셋 폴더가 없습니다: {DATASET_DIR}")
        return 1

    label_files = find_label_files(DATASET_DIR)

    if not label_files:
        print("labels 폴더에서 txt 라벨 파일을 찾지 못했습니다.")
        return 1

    changed_count = 0

    for label_file in label_files:
        if convert_class_id(label_file):
            changed_count += 1

    print("=" * 60)
    print(f"데이터셋 경로: {DATASET_DIR}")
    print(f"검색한 라벨 파일 수: {len(label_files)}")
    print(f"수정된 라벨 파일 수: {changed_count}")
    print(f"클래스 ID 변경: {OLD_CLASS_ID} -> {NEW_CLASS_ID}")
    print("완료되었습니다.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())