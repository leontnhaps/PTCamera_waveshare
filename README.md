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

---

## ✨ 주요 기능

### 1. **스캔 모드 (Scan)**
- 지정된 각도 범위를 자동으로 스캔하여 이미지 수집
- YOLO 결과를 CSV로 자동 로깅 (pan/tilt 각도, 좌표, confidence)
- 스캔 중 흔들림 제어 (Accel, Settle 파라미터 조정)

### 2. **수동 제어 (Manual/LED)**
- Pan/Tilt 각도 직접 조정
- LED 밝기 제어
- 속도 및 가속도 설정

### 3. **프리뷰 & 설정 (Preview & Settings)**
- 실시간 라이브 프리뷰
- 해상도 조정 (640x360 ~ 2592x1944)
- FPS 및 품질 설정
- 왜곡 보정 적용/해제
- Alpha/Balance 파라미터 조정

### 4. **Pointing (자동 타겟팅)**
- 스캔 CSV 데이터 기반 선형 피팅
- 가중 평균으로 타겟 각도 계산
- 실시간 센터링 (YOLO centroid 기반)

### 5. **하드웨어 테스트 (GPIO.py)**
- Raspberry Pi GPIO 핀 제어
- ESP32와 통신하여 LED 제어
- 1초 주기 HIGH/LOW 신호 테스트

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

# 4. 리팩토링된 GUI 실행
cd Refactoring
python Com_main.py
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
cd Refactoring
python Com_main.py
```

### 2. 스캔 및 타겟팅 프로세스

1. **카메라 보정 파일 로드**
   - `Preview & Settings` 탭에서 `Load calib.npz` 클릭

2. **YOLO 모델 로드**
   - `Load YOLO Weights (.pt)` 버튼 클릭
   - 또는 `config.py`에서 자동 로드 설정

3. **스캔 실행**
   - `Scan` 탭에서 Pan/Tilt 범위 및 Step 설정
   - `Start Scan` 클릭
   - 자동으로 CSV 로깅 시작

4. **Pointing 계산**
   - `Pointing` 탭에서 `Select CSV` 버튼으로 스캔 CSV 선택
   - `가중평균 계산` 클릭 → 타겟 각도 계산
   - `Move to Target` 클릭 → 카메라 이동

5. **센터링 (선택)**
   - `Centering Enable` 체크
   - YOLO centroid 기반 실시간 미세 조정

### 3. GPIO 테스트 (하드웨어 검증)

```bash
# Raspberry Pi에서 실행
python3 GPIO.py

# 예상 출력:
# Signal: HIGH (LED ON)
# Signal: LOW (LED OFF)
# (1초 주기 반복)
```

---

## 📂 디렉토리 구조

```
PTCamera_waveshare/
├── 📁 Refactoring/              ✨ 리팩토링된 코드 (권장)
│   ├── Com_main.py              # GUI 메인 파일
│   ├── config.py                # 설정 중앙화
│   ├── network.py               # 네트워크 클라이언트
│   ├── gui_panels.py            # UI 패널
│   ├── processors/              # 이미지 처리 모듈
│   │   ├── undistort_processor.py
│   │   └── yolo_processor.py
│   ├── controllers/             # 비즈니스 로직
│   │   ├── pointing_controller.py
│   │   ├── scan_controller.py
│   │   └── centering_controller.py
│   └── utils/                   # 유틸리티
│       └── geometry.py
│
├── 📁 Com/                      # 레거시 GUI (백업용)
│   └── test.py
│
├── 📁 Server/                   # 중계 서버
│   └── test.py
│
├── 📁 Raspberrypi/              # 라즈베리파이 에이전트
│   └── test.py
│
├── 📄 calib.npz                 # 카메라 보정 파일
├── 📄 yolov11m.pt               # YOLO 모델 (v11 medium)
├── 📄 yolov11s.pt               # YOLO 모델 (v11 small)
├── 📄 GPIO.py                   # GPIO 테스트 코드
└── 📄 README.md
```

---

## 🔧 하드웨어 구성

### 필수 장비

| 항목 | 모델 | 용도 |
|------|------|------|
| **카메라 모듈** | Waveshare 2-Axis Pan-Tilt Camera | 팬-틸트 제어 |
| **메인 컴퓨터** | Raspberry Pi 4 (4GB+) | 카메라 제어 및 이미지 전송 |
| **모터 컨트롤러** | ESP32 | Pan-Tilt 모터 제어 |
| **GUI 클라이언트** | Windows PC (CUDA 지원 권장) | YOLO 처리 및 UI |
| **반사판** | 고휘도 적색 반사 필름 | 객체 인식 타겟 |

### 핀 연결

**Raspberry Pi ↔ ESP32**
- GPIO 15 (BCM) → ESP32 입력 핀
- GND → GND

**ESP32 ↔ Pan-Tilt Motor**
- UART TX/RX → 모터 시리얼 통신

---

## 📅 개발 이력

### **2025-11-26: GPIO 하드웨어 테스트**
- ✅ Raspberry Pi GPIO 15번 핀으로 ESP32 제어
- ✅ LED ON/OFF 1초 주기 테스트 코드 작성

### **2025-11-18: 대규모 리팩토링 완료**
- ✅ **코드 품질 개선**: Com_main.py 997줄 → 792줄 (21% 감소)
- ✅ **모듈화**: controllers/, processors/, utils/ 분리
- ✅ **개발 편의성**: config.py 중앙화, calib/yolo 자동 로드
- ✅ **아키텍처 개선**: 관심사 분리, 재사용성 향상

### **2025-10-27: 리팩토링 계획 수립**
- 디렉토리 정리 완료
- 서버 IP 자동 확인 계획
- 실제 운영용/테스트용 코드 분리 계획

### **2025-09-20: YOLO 학습 개선**
- 보정된 이미지 기반으로 YOLO 재학습
- Dataset_4 이상 학습 진행

### **2025-09-17: YOLO 통합**
- Dataset_3까지 학습한 모델 적용
- 인식 성능 확인 (전반적으로 양호)

### **2025-09-15: 프로젝트 시작**
- ✅ GUI 기능 구현 (Scan, Manual, Preview)
- ✅ GPU 가속 지원 (CUDA)
- ✅ 왜곡 보정 적용
- ✅ 스캔 파라미터 최적화 (흔들림 제어)

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

### 계획된 기능
- [ ] 레이저 트래킹 프로세서 추가
- [ ] 다중 타겟 우선순위 알고리즘
- [ ] 웹 기반 모니터링 대시보드
- [ ] 자동 캘리브레이션 기능

---

## 📄 라이선스

이 프로젝트는 교육 및 연구 목적으로 개발되었습니다.

---

## 👤 개발자

**leontnhaps**
- GitHub: [@leontnhaps](https://github.com/leontnhaps)
- Repository: [PTCamera_waveshare](https://github.com/leontnhaps/PTCamera_waveshare)

---

## 🙏 감사의 말

- [Waveshare](https://www.waveshare.com/) - Pan-Tilt 카메라 모듈
- [Ultralytics](https://github.com/ultralytics/ultralytics) - YOLOv11
- [OpenCV](https://opencv.org/) - 컴퓨터 비전 라이브러리
- [Raspberry Pi Foundation](https://www.raspberrypi.org/) - Picamera2

---

<p align="center">
  <i>광학 무선 전력 전송 시스템을 위한 자동 타겟팅 솔루션</i>
</p>

<p align="center">
  Made with ❤️ for Optical WPT Research
</p>
