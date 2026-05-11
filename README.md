# AI 기반 산업 설비 진동 FFT 분석 (중간 점검 버전)

이 프로젝트는 **AI Hub 기계시설물 고장 예지 센서 데이터셋** 중 `vibration` 데이터만 사용하여,
다음 파이프라인을 수행합니다.

- 데이터셋 폴더 구조 점검
- CSV 로드 및 특정 진동 컬럼 추출
- 전처리(결측치 제거, 평균 제거, 선택적 정규화, 길이 제한)
- FFT 계산 및 지배 주파수(Dominant Frequency) 확인
- 특징값 추출(RMS, 피크값, 상위 주파수 성분, 주파수 대역 에너지)
- 특징 CSV 누적 저장
- 특징 CSV 기반 머신러닝 분류 모델 학습/평가/예측(Random Forest, SVM, MLP)
- 시간영역/주파수영역 그래프 저장

> 현재 버전은 **FFT + 특징 추출 + 특징 기반 머신러닝 분류**까지 구현되어 있습니다.

---

## 1) 폴더 구조

```text
project/
├── data/
│   └── raw/
├── outputs/
│   ├── time_plots/
│   ├── fft_plots/
│   └── logs/
├── src/
│   ├── config.py
│   ├── dataset_inspector.py
│   ├── data_loader.py
│   ├── preprocess.py
│   ├── fft_analysis.py
│   ├── visualize.py
│   └── main.py
├── requirements.txt
└── README.md
```

---

## 2) 환경 설정

`project` 폴더 기준:

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

---

## 3) 데이터 준비

AI Hub에서 받은 진동 CSV 파일을 아래 경로에 배치하세요.

```text
project/data/raw/vibration/
```

현재 구현은 `vibration` 경로만 처리하도록 제한되어 있습니다.

---

## 4) 실행 방법

아래 예시는 `project/src` 폴더에서 실행한다고 가정합니다.

```bash
cd src
```

### (A) 데이터셋 구조 점검 모드

```bash
python main.py --inspect --folder ../data/raw/vibration
```

출력 내용:
- CSV 파일 개수
- 파일 목록 일부
- 샘플 파일의 컬럼명, shape, 상위 5행

### (B) 단일 파일 FFT 분석 모드

```bash
python main.py --file ../data/raw/vibration/sample.csv --column acc_x --fs 1000
```

### (C) 폴더 내 다중 파일 FFT 분석 모드

```bash
python main.py --folder ../data/raw/vibration --column acc_x --fs 1000 --limit 5
```

### (D) 추가 옵션

- `--max-samples 5000` : 앞부분 샘플만 사용
- `--normalize` : z-score 정규화 적용
- `--auto-fs` : 파일 헤더의 `Sample Rate` 값을 우선 사용
- `--no-plot` : 그래프 저장 생략(대용량 배치 처리 속도 향상)
- `--feature-csv ../outputs/features/features.csv` : 특징 저장 위치 지정
- `--top-k 3` : 상위 주요 주파수 성분 개수 지정

예시:

```bash
python main.py --file ../data/raw/vibration/sample.csv --column acc_x --fs 1000 --max-samples 5000 --normalize --auto-fs
```

전체 데이터셋 특징 추출(그래프 생략):

```bash
python main.py --folder ../data/raw/vibration --column acc_x --auto-fs --no-plot --feature-csv ../outputs/features/features_all.csv
```

---

## 5) 결과물

분석 후 출력 파일:

- 시간영역 그래프: `project/outputs/time_plots/*.png`
- FFT 그래프: `project/outputs/fft_plots/*.png`
- 특징 CSV: `project/outputs/features/*.csv`
- 실행 로그: `project/outputs/logs/analysis_log.txt`

콘솔 출력:
- 사용 컬럼
- 샘플링 주파수
- 샘플 수
- RMS
- Peak(|x|)
- Dominant Frequency (Hz)
- Dominant Amplitude

---

## 6) 예외 처리(주요)

다음 케이스에 대해 명확한 오류 메시지를 출력합니다.

- 빈 CSV 파일
- 잘못된 파일 경로
- 존재하지 않는 컬럼명
- 숫자형이 아닌 컬럼
- 잘못된 샘플링 주파수(`--fs <= 0`)
- 잘못된 제한값(`--limit <= 0`, `--max-samples <= 0`)

---

## 7) 머신러닝(특징 CSV 기반 상태 분류) 실행 방법

머신러닝 단계는 **원시 진동 신호를 다시 처리하지 않고**,  
이미 생성된 `features_all.csv` / `features_all_labeled.csv`를 입력으로 사용합니다.

### (A) features_all_labeled.csv 생성(라벨 빌드)

`file_path`에서 상태 폴더명을 읽어 `label` 컬럼을 생성합니다.

```bash
python label_builder.py --input ../outputs/features/features_all.csv --output ../outputs/features/features_all_labeled.csv --drop-unknown
```

추가로 아래 파일이 생성됩니다.
- `../outputs/ml/label_check_summary.txt`
- `../outputs/ml/class_distribution.csv`

### (B) multiclass 학습

```bash
python train_ml.py --features ../outputs/features/features_all_labeled.csv --label-column label --task multiclass
```

모델 선택 실행(시간 단축):

```bash
python train_ml.py --features ../outputs/features/features_all_labeled.csv --label-column label --task multiclass --models random_forest
python train_ml.py --features ../outputs/features/features_all_labeled.csv --label-column label --task multiclass --models random_forest mlp
python train_ml.py --features ../outputs/features/features_all_labeled.csv --label-column label --task multiclass --models random_forest svm mlp
```

### (C) binary 학습(정상 vs 고장)

```bash
python train_ml.py --features ../outputs/features/features_all_labeled.csv --label-column label --task binary
```

### (D) 교차검증(선택)

```bash
python train_ml.py --features ../outputs/features/features_all_labeled.csv --label-column label --task multiclass --cv 5
```

### (E) Group Split 검증(선택)

```bash
python train_ml.py --features ../outputs/features/features_all_labeled.csv --label-column label --task multiclass --group-split
```

### (F) 예측 실행

```bash
python predict_ml.py --input ../outputs/features/features_all_labeled.csv
python predict_ml.py --input ../outputs/features/features_all_labeled.csv --model ../models/best_model_multiclass.pkl
```

### (G) 결과 파일 설명

- 학습/검증 요약:
  - `../outputs/ml/feature_check_summary.txt`
  - `../outputs/ml/used_features.txt`
  - `../outputs/ml/excluded_features.txt`
  - `../outputs/ml/train_test_split_summary.txt`
  - `../outputs/ml/ml_result_summary_for_report.txt`
- 모델 성능:
  - `../outputs/ml/model_comparison_multiclass.csv`
  - `../outputs/ml/model_comparison_binary.csv`
  - `../outputs/ml/classification_report_*_multiclass.txt`
  - `../outputs/ml/classification_report_*_binary.txt`
  - `../outputs/ml/confusion_matrix_*_multiclass.png`
  - `../outputs/ml/confusion_matrix_*_binary.png`
  - `../outputs/ml/feature_importance_random_forest_multiclass.png`
- 선택 기능 결과:
  - `../outputs/ml/cross_validation_results_multiclass.csv`
  - `../outputs/ml/cross_validation_results_binary.csv`
  - `../outputs/ml/model_comparison_multiclass_group_split.csv`
  - `../outputs/ml/group_split_summary.txt`
- 모델 아티팩트:
  - `../models/best_model_multiclass.pkl`
  - `../models/best_model_binary.pkl`
  - `../models/scaler_multiclass.pkl`, `../models/scaler_binary.pkl`
  - `../models/label_encoder_multiclass.pkl`, `../models/label_encoder_binary.pkl`
  - `../models/best_model_name.txt`

### (H) 주의사항

- `label`(또는 지정한 라벨 컬럼)이 반드시 필요합니다.
- `file_path`, `label`, `column`, `fs_hz`, `samples` 같은 메타 컬럼은 기본적으로 학습에서 제외됩니다.
- Random Forest 성능이 매우 높게 나올 경우, `--group-split`으로 추가 검증해 과대평가 가능성을 확인하는 것을 권장합니다.

## 8) 향후 확장 계획

- 클래스별 데이터 수집 균형화(불균형 완화)
- 하이퍼파라미터 탐색(GridSearchCV/RandomizedSearchCV)
- 주기적 재학습 및 로봇 실측 데이터 온라인 평가 자동화
