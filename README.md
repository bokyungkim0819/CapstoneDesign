# AI 기반 산업 설비 진동 FFT 분석 (중간 점검 버전)

이 프로젝트는 **AI Hub 기계시설물 고장 예지 센서 데이터셋** 중 `vibration` 데이터만 사용하여,
다음 파이프라인을 수행합니다.

- 데이터셋 폴더 구조 점검
- CSV 로드 및 특정 진동 컬럼 추출
- 전처리(결측치 제거, 평균 제거, 선택적 정규화, 길이 제한)
- FFT 계산 및 지배 주파수(Dominant Frequency) 확인
- 특징값 추출(RMS, 피크값, 상위 주파수 성분, 주파수 대역 에너지)
- 특징 CSV 누적 저장(추후 머신러닝 입력으로 사용 가능)
- 시간영역/주파수영역 그래프 저장

> 현재 버전은 **FFT + 특징 추출까지** 구현되어 있으며, 머신러닝/분류/예측 코드는 포함하지 않습니다.

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

## 7) 향후 확장 계획

현재 코드는 향후 머신러닝 기반 상태 진단으로 확장하기 쉽도록 모듈화되어 있습니다.

- `data_loader.py` / `preprocess.py`: 학습용 입력 생성 단계로 확장 가능
- `fft_analysis.py` + `feature_extraction.py`: 주파수 특징량 확장 가능
- `main.py`: 모델 추론/평가 모드 추가 가능
