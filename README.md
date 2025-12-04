# 🎯 PTCamera_waveshare: Optical WPT Targeting System

**Waveshare Pan-Tilt 카메라 모듈을 활용한 광학 무선 전력 전송(Optical WPT) 자동 타겟팅 시스템**

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.x-green.svg)](https://opencv.org/)
[![YOLO](https://img.shields.io/badge/YOLO-v11-red.svg)](https://github.com/ultralytics/ultralytics)

---

## 📋 프로젝트 개요

이 프로젝트는 **Optical Wireless Power Transfer (OWPT)** 시스템을 위한 자동 정밀 타겟팅 솔루션입니다. 컴퓨터 비전(OpenCV)과 딥러닝(YOLO)을 결합하여 반사판(Receiver)을 실시간으로 탐지하고, 레이저를 정확하게 조준하여 전력 전송 효율을 극대화합니다.

### 🌟 핵심 목표
- **자동화 (Automation)**: 스캔부터 정밀 조준까지 원클릭 실행
- **정확성 (Precision)**: 왜곡 보정 및 정밀 피드백 제어
- **속도 (Speed)**: GPU 가속 및 하이브리드 탐지 알고리즘 적용

---

## ⚙️ 시스템 아키텍처

Windows PC(클라이언트)가 중앙 제어를 담당하며, 중계 서버를 통해 Raspberry Pi(에이전트)와 통신하여 Pan-Tilt 모듈을 제어합니다.

```mermaid
graph LR
    Client[🖥️ GUI Client\n(Windows PC)] <-->|Socket| Server[📡 Relay Server]
    Server <-->|Socket| Pi[🍓 Raspberry Pi\n(Agent)]
    Pi <-->|UART/GPIO| PT[📷 Pan-Tilt Camera\n(ESP32/Motor)]
```

---

## ✨ 주요 기능

### 1. 🔄 Auto Sequence (자동 시퀀스)
버튼 하나로 전체 타겟팅 과정을 자동으로 수행합니다.
1. **Scan**: 설정된 범위를 스캔하며 이미지 수집 및 객체 탐지
2. **Compute**: 탐지된 데이터를 분석하여 최적의 타겟 좌표 계산 (선형 회귀)
3. **Move**: 계산된 타겟 위치로 고속 이동
4. **Pointing**: 레이저를 켜고 미세 조정하여 정밀 타겟팅 완료

### 2. 📷 Scan Mode (스캔 모드)
- **Hybrid Detection**: 전체 이미지와 분할(Tiling) 이미지를 동시에 분석하여 크고 작은 객체를 모두 놓치지 않고 탐지
- **Real-time Logging**: 스캔과 동시에 CSV 데이터 기록 및 저장
- **Optimization**: GPU 메모리 상태에 따른 배치 처리 자동 최적화

### 3. 🎯 Pointing Mode (정밀 타겟팅)
타겟 위치로 이동 후, 레이저와 타겟의 오차를 실시간으로 보정합니다.
- **Laser Detection**: 차분 이미지(Difference Image)와 밝기 무게중심(Moments)을 이용한 고속 레이저 위치 검출
- **Feedback Loop**: 레이저 위치와 타겟(YOLO 검출) 사이의 오차를 계산하여 Pan-Tilt 미세 조정
- **Robustness**: 검출 실패 시 무작위 이동 없이 제자리에서 재시도(Retry)하여 안정성 확보
- **Debug Visualization**: 레이저 및 타겟 검출 과정을 시각화한 디버그 이미지 자동 저장

### 4. 🔧 Image Processing (이미지 처리)
- **Undistortion**: 카메라 렌즈 왜곡을 실시간으로 보정 (Calibration 데이터 기반)
- **Difference Imaging**: LED On/Off 이미지를 차분하여 주변광 노이즈 제거 및 반사판 식별력 강화

---

## 🚀 설치 및 실행 방법

### 1. 환경 설정

**Windows (GUI Client)**
- Python 3.8 이상
- CUDA Toolkit (NVIDIA GPU 사용 시 권장)
```bash
pip install opencv-python opencv-contrib-python numpy pillow ultralytics torch torchvision
```

**Raspberry Pi (Agent)**
- Python 3.8 이상
- Picamera2, RPi.GPIO
```bash
sudo apt install python3-picamera2
pip3 install pyserial
```

### 2. 실행 순서

시스템은 **Server → Agent → Client** 순서로 실행해야 합니다.

**Step 1: 중계 서버 실행**
```bash
cd Server
python test.py
```

**Step 2: Raspberry Pi 에이전트 실행**
```bash
# 라즈베리파이에서 실행
cd Raspberrypi
python3 test.py
```

**Step 3: GUI 클라이언트 실행**
```bash
# Windows PC에서 실행
cd Refactoring/Com
python Com_main.py
```

---

## 📂 디렉토리 구조

```
PTCamera_waveshare/
├── 📁 Refactoring/              # ✨ 메인 소스 코드 (리팩토링 완료)
│   ├── 📁 Com/                  # Windows GUI 클라이언트
│   │   ├── Com_main.py          # 메인 실행 파일
│   │   └── test.py              # (Com_main.py와 동일)
│   ├── 📁 Raspberrypi/          # Raspberry Pi 에이전트 코드
│   └── 📁 Server/               # 중계 서버 코드
│
├── 📁 Experiments/              # 🧪 실험 및 유틸리티 도구
│   ├── undistort_gui.py         # 왜곡 보정 테스트 도구
│   ├── generate_diff_dataset.py # 학습 데이터셋 생성기
│   └── ...
│
├── 📄 calib.npz                 # 카메라 캘리브레이션 데이터
├── 📄 yolov11m_diff.pt          # YOLO 학습 모델 가중치
└── 📄 README.md                 # 프로젝트 문서
```

---

## 📄 라이선스

이 프로젝트는 교육 및 연구 목적으로 개발되었습니다.

---

## 👤 개발자

**leontnhaps**
- GitHub: [@leontnhaps](https://github.com/leontnhaps)
- Repository: [PTCamera_waveshare](https://github.com/leontnhaps/PTCamera_waveshare)