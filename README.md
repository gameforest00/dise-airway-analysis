# DISE Airway Analysis System

수면 내시경(DISE) 영상 기반 기도 폐쇄 자동 분석 시스템

## 프로젝트 개요

폐쇄성 수면 무호흡증(OSA) 환자의 약물유도수면내시경(DISE) 영상을 분석하여 기도 폐쇄 부위를 자동으로 탐지하고 분류하는 AI 시스템입니다.

### 주요 기능

- Lumen 영역 자동 탐지 (HSV + Grayscale)
- AI 기반 폐쇄 부위 분류 (EfficientNet-V2)
- 3단계 기도 상태 분류 (Open / Partial / Close)
- HTML 리포트 자동 생성

## 설치
```bash
git clone https://github.com/gameforest00/dise-airway-analysis.git
cd dise-airway-analysis/final_pre
pip install -r requirements.txt
```

## 실행 방법

### Step 1: 데이터 준비
```bash
python scripts/step1_prepare_full.py
```

- DISE 비디오에서 프레임 추출
- ROI 자동 검출 및 전처리
- Train/Validation 분할

### Step 2: 모델 학습
```bash
python scripts/step2_train_pytorch.py
```

- EfficientNet-V2 기반 학습
- Multi-Input (Image + Phase)
- 30 epochs, batch size 32

### Step 3: 비디오 분석
```bash
python scripts/step3_enhanced.py [video_path]
```

또는 웹 인터페이스:
```bash
python scripts/app.py
# http://localhost:5000 접속
```

## AI 모델

### 출력 클래스

- 0: No obstruction
- 1: Velum
- 2: Oropharynx
- 3: Tongue base
- 4: Epiglottis

### 상태 분류 기준

- Open: Lumen 면적 75% 이상
- Partial: Lumen 면적 50-75%
- Close: Lumen 면적 50% 미만

## 결과물
```
results_full/[video_name]/
├── report.html       # 분석 리포트
├── timeline.png      # 타임라인 그래프
└── frames/           # 대표 프레임
```

## 주의사항

- 데이터셋은 개인정보 보호를 위해 미포함
- 연구 및 교육 목적으로만 사용
- 임상 진단 전 전문의 검증 필수
