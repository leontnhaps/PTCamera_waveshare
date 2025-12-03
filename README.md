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
- [하드웨어 구성](#-하드웨어-구성)
- [개발 이력](#-개발-이력)
- [향후 개선 사항](#-향후-개선-사항)
- [기여하기](#-기여하기)

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
- **Workflow**: Scan → Compute Target → Move → Centering → Pointing
- 각 단계별 자동 전환 및 에러 처리

### 2. **스캔 모드 (Scan)**
- 지정된 각도 범위를 자동으로 스캔하여 이미지 수집
- YOLO 결과를 CSV로 자동 로깅 (pan/tilt 각도, 좌표, confidence)
- 스캔 중 흔들림 제어 (Accel, Settle 파라미터 조정)

### 3. **Pointing Mode (정밀 타겟팅)**
- **SAHI (Slicing Aided Hyper Inference) 기법 적용**:
    - 고해상도 이미지를 6등분(2x3) 타일링하여 YOLO 추론
    - 작은 객체(Small Object) 인식률 대폭 향상
- **레이저 트래킹**:
    - LED ON/OFF 차분(Diff) 이미지 분석
    - ROI, GaussianBlur, Threshold, FindContours 기법 활용
    - 레이저 포인트 자동 감지 및 타겟 정렬

### 4. **Centering Mode (중앙 정렬)**
- YOLO Centroid 기반 실시간 피드백 제어
- 목표 객체를 화면 중앙에 유지하도록 Pan/Tilt 미세 조정

### 5. **프리뷰 & 설정 (Preview & Settings)**
- 실시간 라이브 프리뷰
- 해상도 조정 (640x360 ~ 2592x1944)
- FPS 및 품질 설정
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

### 2. Auto Sequence (자동 모드)

1. **준비**: `Load calib.npz` 및 `Load YOLO` 완료
2. **실행**: `Auto Sequence` 탭의 `START AUTO SEQUENCE` 버튼 클릭
3. **동작**:
    - **Scan**: 전체 영역 스캔 및 CSV 저장
    - **Compute**: 타겟 위치 계산
    - **Move**: 타겟 위치로 이동
    - **Centering**: 정밀 중앙 정렬
    - **Pointing**: 레이저 ON/OFF 및 타겟 조준

### 3. 수동 제어 및 테스트

- **Manual Tab**: Pan/Tilt 직접 제어, LED/Laser ON/OFF
- **Preview Tab**: 실시간 화면 확인, 왜곡 보정 설정
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
├── 📁 Docs/                     # 📄 문서 및 논문
│
- ✅ 독립적인 왜곡 보정 도구 개발
- ✅ 실시간 보정 프리뷰 기능 추가

### **2025-11-26: GPIO 하드웨어 테스트**
- ✅ Raspberry Pi GPIO 15번 핀으로 ESP32 제어
- ✅ LED ON/OFF 1초 주기 테스트 코드 작성

### **2025-11-18: 대규모 리팩토링 완료**
- ✅ **코드 품질 개선**: Com_main.py 최적화
- ✅ **모듈화**: controllers/, processors/, utils/ 분리

### **2025-09-15: 프로젝트 시작**
- ✅ GUI 기능 구현 (Scan, Manual, Preview)
- ✅ GPU 가속 지원 (CUDA)
- ✅ 왜곡 보정 적용

---

## 🔬 향후 개선 사항

### 연구 필요
1. **레이저 조준 알고리즘**
   - 1차 조준 후 레이저 깜빡임으로 픽셀 차이 보정
   - 피드백 루프 기반 정밀도 향상

2. **다중 객체 인식**
   - 여러 개의 반사판 동시 추적
   - 우선순위 기반 타겟 선택

### 알려진 문제
- ⚠️ **거리 제한**: 먼 거리에서 반사 성능 저하
- ⚠️ **각도 민감도**: 스캔 각도에 따라 인식률 변동

---

## 📄 라이선스

이 프로젝트는 교육 및 연구 목적으로 개발되었습니다.

---

## 👤 개발자

**leontnhaps**
- GitHub: [@leontnhaps](https://github.com/leontnhaps)
- Repository: [PTCamera_waveshare](https://github.com/leontnhaps/PTCamera_waveshare)

---

<p align="center">
  <i>광학 무선 전력 전송 시스템을 위한 자동 타겟팅 솔루션</i>
</p>

<p align="center">
  Made with ❤️ for Optical WPT Research
</p>
