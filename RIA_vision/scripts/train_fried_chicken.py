#!/usr/bin/env python3
"""
Batter / Done YOLO11 Object Detection 학습 및 테스트 평가 스크립트

클래스:
    0: batter
    1: done

실행:
    python3 ~/ria/RIA_food_ws/scripts/train_fried_chicken.py
"""

from datetime import datetime
import json
from pathlib import Path
import sys

from ultralytics import YOLO


# ============================================================
# 사용자 설정
# ============================================================

WORKSPACE = Path.home() / "ria" / "RIA_food_ws"

DATASET_DIR = (
    WORKSPACE
    / "datasets"
    / "fried_chicken_stage_detection"
)

DATA_YAML = DATASET_DIR / "data.yaml"

# 직접 지정한 모델이 있으면 사용하고,
# 없으면 Ultralytics가 yolo11s.pt를 내려받거나 캐시에서 사용합니다.
MODEL_PATH = WORKSPACE / "yolo11s.pt"
FALLBACK_MODEL = "yolo11s.pt"

PROJECT_DIR = WORKSPACE / "results"

# 매 실행마다 고유 폴더 생성
TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
RUN_NAME = f"batter_done_640_{TIMESTAMP}"
TEST_RUN_NAME = f"{RUN_NAME}_test"

EPOCHS = 100
IMAGE_SIZE = 640
BATCH_SIZE = 4
DEVICE = 0
PATIENCE = 307
LEARNING_RATE = 0.003
WORKERS = 4

IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp",
}


def count_files(directory: Path, extensions: set[str]) -> int:
    """지정한 확장자의 파일 수를 계산합니다."""
    if not directory.exists():
        return 0

    return sum(
        1
        for path in directory.iterdir()
        if path.is_file()
        and path.suffix.lower() in extensions
    )


def inspect_label_classes(label_dir: Path) -> dict[int, int]:
    """라벨 폴더 안의 클래스별 객체 수를 계산합니다."""
    class_counts: dict[int, int] = {}

    for label_path in label_dir.glob("*.txt"):
        with label_path.open(
            "r",
            encoding="utf-8",
            errors="replace",
        ) as file:
            for line_number, line in enumerate(file, start=1):
                stripped = line.strip()

                if not stripped:
                    continue

                parts = stripped.split()

                if len(parts) != 5:
                    raise ValueError(
                        "Detection 라벨 형식이 아닙니다: "
                        f"{label_path}:{line_number}\n"
                        f"값 개수: {len(parts)}"
                    )

                try:
                    class_id = int(parts[0])
                    coordinates = [
                        float(value)
                        for value in parts[1:]
                    ]
                except ValueError as error:
                    raise ValueError(
                        "라벨 숫자 변환에 실패했습니다: "
                        f"{label_path}:{line_number}"
                    ) from error

                if class_id not in (0, 1):
                    raise ValueError(
                        f"허용되지 않은 클래스 ID {class_id}: "
                        f"{label_path}:{line_number}"
                    )

                if not all(
                    0.0 <= value <= 1.0
                    for value in coordinates
                ):
                    raise ValueError(
                        "라벨 좌표가 0~1 범위를 벗어났습니다: "
                        f"{label_path}:{line_number}"
                    )

                class_counts[class_id] = (
                    class_counts.get(class_id, 0) + 1
                )

    return class_counts


def inspect_image_label_pairs(
    image_dir: Path,
    label_dir: Path,
) -> tuple[set[str], set[str]]:
    """이미지와 라벨 파일명이 서로 대응하는지 확인합니다."""
    image_stems = {
        path.stem
        for path in image_dir.iterdir()
        if path.is_file()
        and path.suffix.lower() in IMAGE_EXTENSIONS
    }

    label_stems = {
        path.stem
        for path in label_dir.glob("*.txt")
        if path.is_file()
    }

    images_without_labels = image_stems - label_stems
    labels_without_images = label_stems - image_stems

    return images_without_labels, labels_without_images


def inspect_dataset() -> None:
    """train, valid, test 구조와 라벨을 점검합니다."""
    print("=" * 70)
    print("데이터셋 점검")
    print("=" * 70)
    print(f"데이터셋 경로: {DATASET_DIR}")
    print(f"설정 파일: {DATA_YAML}")
    print()

    if not DATASET_DIR.exists():
        raise FileNotFoundError(
            f"데이터셋 폴더가 없습니다: {DATASET_DIR}"
        )

    if not DATA_YAML.exists():
        raise FileNotFoundError(
            f"data.yaml이 없습니다: {DATA_YAML}"
        )

    total_images = 0
    total_labels = 0
    total_class_counts = {
        0: 0,
        1: 0,
    }

    for split in ("train", "valid", "test"):
        image_dir = DATASET_DIR / split / "images"
        label_dir = DATASET_DIR / split / "labels"

        if not image_dir.exists():
            raise FileNotFoundError(
                f"이미지 폴더가 없습니다: {image_dir}"
            )

        if not label_dir.exists():
            raise FileNotFoundError(
                f"라벨 폴더가 없습니다: {label_dir}"
            )

        image_count = count_files(
            image_dir,
            IMAGE_EXTENSIONS,
        )
        label_count = count_files(
            label_dir,
            {".txt"},
        )

        (
            images_without_labels,
            labels_without_images,
        ) = inspect_image_label_pairs(
            image_dir,
            label_dir,
        )

        class_counts = inspect_label_classes(label_dir)

        total_images += image_count
        total_labels += label_count

        for class_id, count in class_counts.items():
            total_class_counts[class_id] += count

        print(f"[{split}]")
        print(f"  images: {image_count}")
        print(f"  labels: {label_count}")
        print(
            f"  batter objects: "
            f"{class_counts.get(0, 0)}"
        )
        print(
            f"  done objects: "
            f"{class_counts.get(1, 0)}"
        )
        print(
            f"  이미지에 대응하는 라벨 누락: "
            f"{len(images_without_labels)}"
        )
        print(
            f"  라벨에 대응하는 이미지 누락: "
            f"{len(labels_without_images)}"
        )

        if images_without_labels:
            examples = sorted(
                images_without_labels
            )[:10]

            raise RuntimeError(
                f"{split} 이미지의 라벨이 누락되었습니다: "
                f"{examples}"
            )

        if labels_without_images:
            examples = sorted(
                labels_without_images
            )[:10]

            raise RuntimeError(
                f"{split} 라벨의 이미지가 누락되었습니다: "
                f"{examples}"
            )

        print()

    print(f"전체 이미지: {total_images}")
    print(f"전체 라벨 파일: {total_labels}")
    print(
        f"전체 batter 객체: "
        f"{total_class_counts[0]}"
    )
    print(
        f"전체 done 객체: "
        f"{total_class_counts[1]}"
    )
    print()
    print("데이터셋 구조와 라벨 검사가 완료되었습니다.")
    print()


def get_initial_model() -> YOLO:
    """초기 YOLO 모델을 불러옵니다."""
    if MODEL_PATH.exists():
        print(f"로컬 모델 사용: {MODEL_PATH}")
        return YOLO(str(MODEL_PATH))

    print(f"기본 사전학습 모델 사용: {FALLBACK_MODEL}")
    return YOLO(FALLBACK_MODEL)


def train_model() -> Path:
    """YOLO11 Detection 모델을 학습합니다."""
    result_dir = PROJECT_DIR / RUN_NAME

    print("=" * 70)
    print("YOLO11 Batter / Done Detection 학습 시작")
    print("=" * 70)
    print(f"클래스 0: batter")
    print(f"클래스 1: done")
    print(f"이미지 크기: {IMAGE_SIZE}")
    print(f"Epochs: {EPOCHS}")
    print(f"Batch: {BATCH_SIZE}")
    print(f"Device: {DEVICE}")
    print(f"결과 폴더: {result_dir}")
    print()

    model = get_initial_model()

    model.train(
        data=str(DATA_YAML),
        epochs=EPOCHS,
        imgsz=IMAGE_SIZE,
        batch=BATCH_SIZE,
        device=DEVICE,
        patience=PATIENCE,
        lr0=LEARNING_RATE,
        workers=WORKERS,
        project=str(PROJECT_DIR),
        name=RUN_NAME,
        exist_ok=False,
        pretrained=True,
        verbose=True,
        plots=True,
    )

    best_model_path = (
        result_dir
        / "weights"
        / "best.pt"
    )

    last_model_path = (
        result_dir
        / "weights"
        / "last.pt"
    )

    if not best_model_path.exists():
        raise FileNotFoundError(
            f"best.pt가 생성되지 않았습니다: "
            f"{best_model_path}"
        )

    print()
    print("=" * 70)
    print("학습 완료")
    print("=" * 70)
    print(f"최적 모델: {best_model_path}")
    print(f"마지막 모델: {last_model_path}")
    print()

    return best_model_path


def evaluate_test_set(best_model_path: Path) -> None:
    """학습된 best.pt를 test split에서 평가합니다."""
    print("=" * 70)
    print("Test 데이터셋 최종 평가 시작")
    print("=" * 70)
    print(f"평가 모델: {best_model_path}")
    print(f"평가 split: test")
    print()

    model = YOLO(str(best_model_path))

    metrics = model.val(
        data=str(DATA_YAML),
        split="test",
        imgsz=IMAGE_SIZE,
        batch=BATCH_SIZE,
        device=DEVICE,
        workers=WORKERS,
        project=str(PROJECT_DIR),
        name=TEST_RUN_NAME,
        exist_ok=False,
        plots=True,
        verbose=True,
    )

    test_result_dir = PROJECT_DIR / TEST_RUN_NAME
    test_result_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    summary = {
        "model": str(best_model_path),
        "dataset": str(DATA_YAML),
        "split": "test",
        "classes": {
            "0": "batter",
            "1": "done",
        },
        "metrics": {
            "precision": float(metrics.box.mp),
            "recall": float(metrics.box.mr),
            "mAP50": float(metrics.box.map50),
            "mAP50_95": float(metrics.box.map),
        },
    }

    json_path = (
        test_result_dir
        / "test_metrics_summary.json"
    )

    text_path = (
        test_result_dir
        / "test_metrics_summary.txt"
    )

    json_path.write_text(
        json.dumps(
            summary,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    text = (
        "Batter / Done YOLO Test Evaluation\n"
        "====================================\n"
        f"Model: {best_model_path}\n"
        f"Dataset: {DATA_YAML}\n"
        "Split: test\n\n"
        f"Precision: {metrics.box.mp:.6f}\n"
        f"Recall: {metrics.box.mr:.6f}\n"
        f"mAP50: {metrics.box.map50:.6f}\n"
        f"mAP50-95: {metrics.box.map:.6f}\n"
    )

    text_path.write_text(
        text,
        encoding="utf-8",
    )

    print()
    print("=" * 70)
    print("Test 평가 완료")
    print("=" * 70)
    print(f"Precision: {metrics.box.mp:.4f}")
    print(f"Recall: {metrics.box.mr:.4f}")
    print(f"mAP50: {metrics.box.map50:.4f}")
    print(f"mAP50-95: {metrics.box.map:.4f}")
    print(f"평가 결과 폴더: {test_result_dir}")
    print(f"JSON 요약: {json_path}")
    print(f"TXT 요약: {text_path}")


def main() -> int:
    try:
        inspect_dataset()

        best_model_path = train_model()

        evaluate_test_set(best_model_path)

        return 0

    except KeyboardInterrupt:
        print(
            "\n사용자에 의해 작업이 중단되었습니다."
        )
        return 130

    except Exception as error:
        print(
            f"\n오류: {error}",
            file=sys.stderr,
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())