"""프로젝트 전역 설정값을 관리하는 모듈."""

from pathlib import Path

PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent
DATA_RAW_DIR: Path = PROJECT_ROOT / "data" / "raw"
OUTPUT_DIR: Path = PROJECT_ROOT / "outputs"
TIME_PLOTS_DIR: Path = OUTPUT_DIR / "time_plots"
FFT_PLOTS_DIR: Path = OUTPUT_DIR / "fft_plots"
LOG_DIR: Path = OUTPUT_DIR / "logs"
FEATURES_DIR: Path = OUTPUT_DIR / "features"

DEFAULT_FS: float = 1000.0

DEFAULT_MAX_SAMPLES: int | None = None

DEFAULT_VIBRATION_COLUMN: str | None = None

DEFAULT_TOP_K_FREQUENCIES: int = 3
DEFAULT_BANDS: list[tuple[float, float]] = [
    (0.0, 10.0),
    (10.0, 30.0),
    (30.0, 60.0),
    (60.0, 120.0),
    (120.0, 250.0),
    (250.0, 500.0),
]


def ensure_output_directories() -> None:
    """출력에 필요한 폴더들을 생성한다."""
    TIME_PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    FFT_PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    FEATURES_DIR.mkdir(parents=True, exist_ok=True)
