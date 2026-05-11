"""FFT 특징 CSV 기반 머신러닝 학습 설정 모듈."""

from __future__ import annotations

from pathlib import Path

from config import PROJECT_ROOT

# 특징 CSV 기본 경로 (학습 스크립트에서 --features로 덮어쓸 수 있음)
DEFAULT_FEATURE_CSV: Path = PROJECT_ROOT / "outputs" / "features" / "features_all_labeled.csv"

# 머신러닝 결과 및 모델 저장 경로
MODEL_DIR: Path = PROJECT_ROOT / "models"
OUTPUT_ML_DIR: Path = PROJECT_ROOT / "outputs" / "ml"

# 학습 시 저장되는 기본 파일 경로(하위 스크립트에서 task 접미사를 붙여 저장 가능)
BEST_MODEL_PATH: Path = MODEL_DIR / "best_model.pkl"
SCALER_PATH: Path = MODEL_DIR / "scaler.pkl"
LABEL_ENCODER_PATH: Path = MODEL_DIR / "label_encoder.pkl"
BEST_MODEL_NAME_PATH: Path = MODEL_DIR / "best_model_name.txt"
MODEL_COMPARISON_PATH: Path = OUTPUT_ML_DIR / "model_comparison.csv"
PREDICTION_OUTPUT_PATH: Path = OUTPUT_ML_DIR / "prediction_results.csv"

# 재현 가능한 결과를 위한 고정 시드
RANDOM_SEED: int = 42

# 학습/검증 분리 비율
TEST_SIZE: float = 0.2

# 기본 라벨 컬럼명
DEFAULT_LABEL_COLUMN: str = "label"

# 학습 대상 모델 목록
MODEL_LIST: list[str] = ["random_forest", "svm", "mlp"]

# 학습에서 제외할 수 있는 메타/경로성 컬럼 후보
DEFAULT_EXCLUDE_COLUMNS: list[str] = [
    "file_name",
    "file_path",
    "path",
    "column",
    "label",
    "status",
    "target",
    "class",
    "filename",
    "source_path",
    # 아래 2개는 숫자형이지만 데이터 조건을 외울 수 있어 기본 제외한다.
    "fs_hz",
    "samples",
]
