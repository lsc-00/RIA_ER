#!/usr/bin/env python3
"""
학습 완료된 fried_chicken_best.pt 모델을 test 데이터셋으로 평가합니다.

실행:
    python3 ~/ria/RIA_food_ws/scripts/evaluate_fried_chicken_test.py

결과 저장:
    ~/ria/RIA_food_ws/results/fried_chicken_test_evaluation/
"""

from __future__ import annotations

from datetime import datetime
import json
from pathlib import Path
import sys

from ultralytics import YOLO


WORKSPACE = Path.home() / "ria" / "RIA_food_ws"
MODEL_PATH = WORKSPACE / "models" / "fried_chicken_best.pt"
DATA_YAML = WORKSPACE / "datasets" / "fried_chicken_detection" / "data.yaml"
PROJECT_DIR = WORKSPACE / "results"
RUN_NAME = "fried_chicken_test_evaluation"

IMAGE_SIZE = 640
BATCH_SIZE = 4
DEVICE = 0
WORKERS = 4
CONFIDENCE_THRESHOLD = 0.001
IOU_THRESHOLD = 0.7


def validate_paths() -> None:
    """평가에 필요한 모델 및 데이터셋 경로를 확인합니다."""
    test_images = WORKSPACE / "datasets" / "fried_chicken_detection" / "test" / "images"
    test_labels = WORKSPACE / "datasets" / "fried_chicken_detection" / "test" / "labels"

    required_paths = {
        "모델 파일": MODEL_PATH,
        "data.yaml": DATA_YAML,
        "test 이미지 폴더": test_images,
        "test 라벨 폴더": test_labels,
    }

    print("=" * 70)
    print("Test 평가 경로 점검")
    print("=" * 70)

    for name, path in required_paths.items():
        print(f"{name:18}: {path}")
        if not path.exists():
            raise FileNotFoundError(f"{name}이(가) 존재하지 않습니다: {path}")

    image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
    image_count = sum(
        1
        for path in test_images.iterdir()
        if path.is_file() and path.suffix.lower() in image_extensions
    )
    label_count = sum(
        1
        for path in test_labels.iterdir()
        if path.is_file() and path.suffix.lower() == ".txt"
    )

    print()
    print(f"Test 이미지 수: {image_count}")
    print(f"Test 라벨 수  : {label_count}")

    if image_count == 0:
        raise RuntimeError("test/images 폴더에 이미지가 없습니다.")
    if label_count == 0:
        raise RuntimeError("test/labels 폴더에 라벨이 없습니다.")
    if image_count != label_count:
        print("경고: Test 이미지 수와 라벨 수가 일치하지 않습니다.", file=sys.stderr)

    print("경로 및 데이터셋 점검 완료")
    print()


def safe_float(value) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def save_summary(output_dir: Path, metrics) -> None:
    """핵심 평가 지표를 TXT와 JSON 파일로 저장합니다."""
    output_dir.mkdir(parents=True, exist_ok=True)

    precision = safe_float(metrics.box.mp)
    recall = safe_float(metrics.box.mr)
    map50 = safe_float(metrics.box.map50)
    map50_95 = safe_float(metrics.box.map)
    fitness = safe_float(metrics.fitness)

    summary = {
        "evaluated_at": datetime.now().isoformat(timespec="seconds"),
        "model_path": str(MODEL_PATH),
        "data_yaml": str(DATA_YAML),
        "split": "test",
        "image_size": IMAGE_SIZE,
        "batch_size": BATCH_SIZE,
        "device": DEVICE,
        "confidence_threshold": CONFIDENCE_THRESHOLD,
        "iou_threshold": IOU_THRESHOLD,
        "precision": precision,
        "recall": recall,
        "mAP50": map50,
        "mAP50_95": map50_95,
        "fitness": fitness,
        "result_directory": str(output_dir),
    }

    json_path = output_dir / "test_metrics_summary.json"
    txt_path = output_dir / "test_metrics_summary.txt"

    with json_path.open("w", encoding="utf-8") as file:
        json.dump(summary, file, ensure_ascii=False, indent=2)

    lines = [
        "=" * 70,
        "Fried Chicken Test Evaluation Summary",
        "=" * 70,
        f"평가 시각       : {summary['evaluated_at']}",
        f"모델 경로       : {MODEL_PATH}",
        f"data.yaml       : {DATA_YAML}",
        "평가 split      : test",
        f"Precision       : {precision:.6f}" if precision is not None else "Precision       : 확인 불가",
        f"Recall          : {recall:.6f}" if recall is not None else "Recall          : 확인 불가",
        f"mAP50           : {map50:.6f}" if map50 is not None else "mAP50           : 확인 불가",
        f"mAP50-95        : {map50_95:.6f}" if map50_95 is not None else "mAP50-95        : 확인 불가",
        f"Fitness         : {fitness:.6f}" if fitness is not None else "Fitness         : 확인 불가",
        f"결과 폴더       : {output_dir}",
        "=" * 70,
    ]

    txt_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print()
    print("\n".join(lines))
    print(f"요약 TXT  : {txt_path}")
    print(f"요약 JSON : {json_path}")


def evaluate_test() -> None:
    """best.pt 모델을 test split으로 평가합니다."""
    print("=" * 70)
    print("YOLO11 Fried Chicken Test 평가 시작")
    print("=" * 70)
    print(f"모델             : {MODEL_PATH}")
    print(f"data.yaml        : {DATA_YAML}")
    print(f"이미지 크기      : {IMAGE_SIZE}")
    print(f"Batch            : {BATCH_SIZE}")
    print(f"Device           : {DEVICE}")
    print(f"결과 저장 위치   : {PROJECT_DIR / RUN_NAME}")
    print()

    model = YOLO(str(MODEL_PATH))

    metrics = model.val(
        data=str(DATA_YAML),
        split="test",
        imgsz=IMAGE_SIZE,
        batch=BATCH_SIZE,
        device=DEVICE,
        workers=WORKERS,
        conf=CONFIDENCE_THRESHOLD,
        iou=IOU_THRESHOLD,
        project=str(PROJECT_DIR),
        name=RUN_NAME,
        exist_ok=True,
        plots=True,
        save_json=False,
        verbose=True,
    )

    output_dir = Path(metrics.save_dir)
    save_summary(output_dir, metrics)
    print("\nTest 데이터셋 평가가 완료되었습니다.")


def main() -> int:
    try:
        validate_paths()
        evaluate_test()
        return 0
    except KeyboardInterrupt:
        print("\n사용자에 의해 평가가 중단되었습니다.")
        return 130
    except Exception as error:
        print(f"\n오류: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
