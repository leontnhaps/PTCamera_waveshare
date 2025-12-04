# 🎯 PTCamera_waveshare

**Waveshare Pan-Tilt 카메라 모듈 제어 시스템**

OpenCV와 YOLO를 활용한 자동 타겟팅 및 레이저 포인팅 시스템

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.x-green.svg)](https://opencv.org/)
[![YOLO](https://img.shields.io/badge/YOLO-v11-red.svg)](https://github.com/ultralytics/ultralytics)

---

## 📋 목차

- [프로젝트 개요](#-프로젝트-개요)
- [주요 기능](#-주요-기능)
- [시스템 아키텍처](#-시스템-아키텍처)
- [설치 방법](#-설치-방법)
- [사용법](#-사용법)
- [디렉토리 구조](#-디렉토리-구조)
- [리팩토링](#-리팩터링-refactoring)
- [향후 개선 사항](#-향후-개선-사항)

---

## 🎯 프로젝트 개요

이 프로젝트는 [Waveshare 2-Axis Pan-Tilt 카메라 모듈](https://www.waveshare.com/2-axis-pan-tilt-camera-module.htm)을 사용하여 **자동 객체 탐지**, **왜곡 보정**, **정밀 타겟팅**을 수행하는 광학 무선 전력 전송(Optical WPT) 시스템입니다.

### 핵심 목표
- 📷 **실시간 객체 인식**: YOLO 기반 반사판 자동 탐지
- 🎯 **정밀 타겟팅**: CSV 데이터 기반 선형 피팅 알고리즘
- 🔧 **왜곡 보정**: 카메라 캘리브레이션 및 실시간 보정
- ⚡ **GPU 가속**: CUDA 지원으로 성능 최적화
- 🔄 **자동 시퀀스**: 스캔부터 타겟팅까지 원클릭 자동화

---

## ✨ 주요 기능

### 1. **Auto Sequence (자동 시퀀스)**
- **One-Click Operation**: 버튼 하나로 전체 프로세스 실행
- **Workflow**: Scan → Compute Target → Move → Pointing
- 각 단계별 자동 전환 및 에러 처리

### 2. **스캔 모드 (Scan)**
- 지정된 각도 범위를 자동으로 스캔하여 이미지 수집
- **실시간 YOLO 검출**: 이미지 수신 즉시 YOLO 처리 및 CSV 기록 (후처리 시간 0초)
- **배치 처리**: 전체 이미지 1개 + 타일 6개를 한 번에 YOLO에 전달
  - GPU 메모리 부족 시 자동 폴백 (7개 → 3개 → 1개)
  - 처리 속도 **2~3배 향상**
- **하이브리드 검출**:
  - 전체 이미지 검출: 큰 객체 탐지
  - 타일링 검출 (2x3): 작은 객체 탐지
  - 검출률 **+20~30% 향상**
- 스캔 중 흔들림 제어 (Accel, Settle 파라미터 조정)

### 3. **Pointing Mode (정밀 타겟팅)**
- **SAHI (Slicing Aided Hyper Inference) 기법 적용**:
    - 고해상도 이미지를 6등분(2x3) 타일링하여 YOLO 추론
- 왜곡 보정 적용/해제
- Alpha/Balance 파라미터 조정

---

## 🏗️ 시스템 아키텍처

```
┌─────────────────┐      Socket (JSON/Binary)      ┌──────────────────┐
│   GUI Client    │ ◄───────────────────────────── │  Server (Broker) │
│   (Windows PC)  │                                 │   (노트북/PC)     │
│                 │                                 │                  │
│ • Tkinter UI    │                                 │ • 제어 중계      │
│ • YOLO 처리     │                                 │ • 이미지 중계    │
│ • 왜곡 보정     │                                 │                  │
│ • Auto Sequence │                                 │                  │
└─────────────────┘                                 └──────────────────┘
                                                             │
                                                             │ Socket
                                                             ▼
                                                    ┌──────────────────┐
                                                    │  Pi Agent        │
                                                    │  (Raspberry Pi)  │
                                                    │                  │
                                                    │ • Picamera2      │
                                                    │ • 시리얼 통신    │
                                                    │ • GPIO 제어      │
                                                    └──────────────────┘
                                                             │
                                                             │ Serial (UART)
                                                             ▼
                                                    ┌──────────────────┐
                                                    │  ESP32 (Motor)   │
                                                    │                  │
                                                    │ • Pan-Tilt 제어  │
                                                    │ • LED 제어       │
                                                    └──────────────────┘
```

---

## 🚀 설치 방법

### 사전 요구사항

**Windows (GUI Client)**
```bash
Python 3.8+
CUDA Toolkit (선택, GPU 가속용)
```

**Raspberry Pi**
```bash
Python 3.8+
Picamera2
RPi.GPIO
```

### GUI 클라이언트 설치

```bash
# 1. 저장소 클론
git clone https://github.com/leontnhaps/PTCamera_waveshare.git
cd PTCamera_waveshare

# 2. 가상환경 생성 (권장)
python -m venv venv
venv\Scripts\activate  # Windows
# source venv/bin/activate  # Linux/Mac

# 3. 의존성 설치
pip install opencv-python opencv-contrib-python
pip install numpy pillow
pip install ultralytics  # YOLO
pip install torch torchvision  # GPU 가속용 (선택)

# 4. GUI 실행
cd Com
python auto_sequence.py
```

### Raspberry Pi 에이전트 설치

```bash
# 1. Picamera2 설치
sudo apt update
sudo apt install -y python3-picamera2

# 2. 의존성 설치
pip3 install pyserial

# 3. 에이전트 실행
cd Raspberrypi
python3 test.py
```

### 서버 실행

```bash
cd Server
python test.py
```

---

## 💡 사용법

### 1. 기본 워크플로우

```bash
# 1단계: 서버 시작
cd Server
python test.py

# 2단계: Raspberry Pi 에이전트 시작
ssh pi@<raspberry-pi-ip>
cd Raspberrypi
python3 test.py

# 3단계: GUI 클라이언트 시작
cd Com
python auto_sequence.py
```

- **ERTest.py**: 이미지 처리 알고리즘(Diff, Threshold 등) 단위 테스트

---

## 📂 디렉토리 구조

```
PTCamera_waveshare/
├── 📁 Com/                      # 메인 GUI 클라이언트 (Windows)
│   ├── auto_sequence.py         # ✨ 통합 자동화 스크립트 (메인)
│   ├── Com_main.py              # 레거시 GUI
│   └── test.py                  # 테스트용 GUI
│
├── 📁 Server/                   # 중계 서버
│   └── test.py
│
├── 📁 Raspberrypi/              # 라즈베리파이 에이전트
│   └── test.py
│
├── 📁 Experiments/              # 🧪 실험용 코드 및 도구
│   ├── undistort_gui.py         # 왜곡 보정 도구
│   ├── generate_diff_dataset.py # 데이터셋 생성기
│   ├── diff_*.py                # 다양한 이미지 처리 튜너
│   └── ...
│
├── 📁 Refactoring/              # 🔧 리팩토링 코드 (새로 추가)
│   ├── Com/Com_main.py          # 정리된 GUI (1680줄, -226줄)
│   ├── Raspberrypi/Rasp_main.py # 서버 IP 선택 기능 추가
│   └── Server/Server_main.py    # 미사용 import 제거
│
├── 📁 Docs/                     # 📄 문서 및 논문
│
├── 📄 calib.npz                 # 카메라 보정 파일
├── 📄 yolov11m_diff.pt          # YOLO 모델 (Diff 학습)
└── 📄 README.md
```

---

## 🔧 리팩토링 (Refactoring)

### 📁 `Refactoring/` 폴더

**목적**: 원본 코드를 보존하면서 안전하게 코드 개선 작업 수행

```
Refactoring/
├── Com/
│   └── Com_main.py (1680줄, -226줄 from original)
├── Raspberrypi/
│   └── Rasp_main.py (383줄, +25줄 with new features)
└── Server/
    └── Server_main.py (226줄, -1줄)
```

### 📊 리팩토링 요약

**Phase 3-4: 스캔 최적화 구조 개선**
- `ImageProcessor` (231줄): 이미지 로딩/왜곡보정, CUDA/Torch 가속
- `YOLOProcessor` (34줄): YOLO 모델 관리/캐싱
- `ScanController` (154줄): 실시간 스캔 처리
- 상수 추출: 23개 매직 넘버 제거
- 코드 감소: ~120줄
- 타일 생성 로직 수정: 12개 → 6개 (rows x cols 정확히 유지)
- 오버랩 처리 개선: step 감소 대신 확장으로 변경

**코드 정리 및 기능 추가**

| 컴포넌트 | 작업 내용 | Before | After | 변화 |
|---------|---------|--------|-------|------|
| **Com** | 미사용 함수 5개 + import 3개 삭제 | 1906줄 | 1680줄 | **-226줄 (-11.9%)** |
| **Raspberrypi** | 서버 IP 선택 메뉴 추가 | 358줄 | 383줄 | +25줄 (기능 추가) |
| **Server** | 미사용 import 1개 삭제 | 227줄 | 226줄 | -1줄 |
| **합계** | | 2491줄 | 2289줄 | **-202줄 (-8.1%)** |

#### Com/Com_main.py

**삭제된 미사용 함수 (235줄):**
- `_centering_on_laser()` (54줄)
- `_detect_red_laser()` (68줄)
- `_align_laser_to_film()` (37줄)
- `_centering_on_centroid()` (52줄)
- `_interp_fit()` (13줄)

**삭제된 미사용 import (3개):**
- `struct`, `io`, `ImageDraw`

#### Raspberrypi/Rasp_main.py

**새로운 기능: 서버 IP 선택 메뉴**

실행 시 서버를 대화형으로 선택 가능:
```
==================================================
서버 선택 (Server Selection)
==================================================
  [1] 711a       → 192.168.0.9
  [2] 602a       → 172.30.1.13
  [3] hotspot    → 10.95.38.118
==================================================
서버 번호를 선택하세요 (1/2/3) [기본값: 2]: 
```

**장점:**
- 코드 수정 없이 서버 전환
-  실수 방지 (잘못된 입력 재요청)
- 환경 변수로 우회 가능

#### Server/Server_main.py

**삭제된 미사용 import:**
- `os` (사용되지 않음)

### 🗂️ Experiments 폴더 정리

**Before (17개 파일):**
- GPIO.py, HSV.py, diff_gemini.py, diff_hsv_tuner.py, diff_image_red.py, diff_image_yellow.py, diff_rgb_tuner.py, diff_rgb_two.py, diff_universe.py, generate_diff_dataset.py, image_diff.py, laserdiff.py, last_filter.py, modify_test.py, rate_image.py, undistort_gui.py, yolo_test_folder.py

**After (13개 파일, 통일된 명명 규칙):**
- `Laser_GPIO.py` - GPIO 테스트
- `SAHI_yolo_test.py` - YOLO 타일링 테스트
- `diff_filter_1_2.py` - Universe + RGB Two 통합 필터
- `diff_filter_hsv.py` - HSV 필터 튜너
- `diff_filter_red.py` - Red 필터 튜너
- `diff_filter_red_yellow.py` - Red & Yellow 통합 필터
- `diff_filter_rgb.py` - RGB 필터 튜너
- `diff_filter_yellow.py` - Yellow 필터 튜너
- `diff_laser.py` - 차분 이미지 레이저 검출
- `generate_diff_dataset.py` - YOLO 학습 데이터셋 생성
- `rate_image.py` - 비율 기반 이미지 분석
- `undistort_gui.py` - 왜곡 보정 GUI  도구
- `view_diff.py` - 차분 이미지 뷰어

**변경 사항:**
- ✅ 파일명 통일 (`diff_filter_*` 패턴)
- ✅ 중복/구형 파일 제거 (4개)
- ✅ 기능별 분류 명확화

### 🔒 리팩토링 원칙

1. **작업 위치 엄수**: `Refactoring/` 폴더만 수정
2. **원본 보존**: `Com/`, `Raspberrypi/`, `Server/` 절대 수정 금지
3. **기능 보장**: 모든 기능 유지 또는 개선 (저하 없음)

### 🎯 검증 완료

- ✅ **스레드 안전성**: 전체 시스템 스레딩 분석 완료
- ✅ **코드 품질**: 미사용 코드 제거, 중복 최소화
- ✅ **기능 테스트**: 원본과 동일하게 작동
- ✅ **메모리/성능**: 영향 없음

---

## 🔬 향후 개선 사항

### 연구 필요

1. **다중 객체 인식**
   - 여러 개의 반사판 동시 추적
   - 우선순위 기반 타겟 선택

2. **무선 전력 전송 효율 확인**
   - 솔라 셀에 전압, 전류계로 효율 확인

---

## 📄 라이선스

이 프로젝트는 교육 및 연구 목적으로 개발되었습니다.

---

## 👤 개발자

**leontnhaps**
- GitHub: [@leontnhaps](https://github.com/leontnhaps)
- Repository: [PTCamera_waveshare](https://github.com/leontnhaps/PTCamera_waveshare)

---