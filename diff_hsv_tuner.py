#!/usr/bin/env python3
"""
LED ON/OFF 차분 이미지에서 HSV 필터링 테스트 도구
image_diff.py + HSV.py 통합 버전
"""

import cv2
import numpy as np
from tkinter import Tk, filedialog
import os

def nothing(x):
    pass

def load_image_with_hangul(image_path):
    """한글 경로를 지원하는 이미지 로드 함수"""
    with open(image_path, 'rb') as f:
        image_array = np.frombuffer(f.read(), dtype=np.uint8)
    img = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
    return img

def select_image(title="이미지 선택"):
    """파일 선택 다이얼로그"""
    root = Tk()
    root.withdraw()
    
    initial_dir = os.path.dirname(os.path.abspath(__file__))
    
    image_path = filedialog.askopenfilename(
        initialdir=initial_dir,
        title=title,
        filetypes=(
            ("이미지 파일", "*.jpg *.jpeg *.png *.bmp"),
            ("모든 파일", "*.*")
        )
    )
    
    root.destroy()
    return image_path

# ==========================================
# 1. 이미지 선택
# ==========================================
print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
print("LED ON/OFF 차분 이미지 HSV 필터링 테스트")
print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n")

print(">> LED ON 이미지를 선택하세요...")
path_on = select_image("1. LED ON 이미지 선택")

if not path_on:
    print("❌ 파일 선택이 취소되었습니다.")
    exit()

print(f"✓ LED ON: {os.path.basename(path_on)}")

print("\n>> LED OFF 이미지를 선택하세요...")
path_off = select_image("2. LED OFF 이미지 선택")

if not path_off:
    print("❌ 파일 선택이 취소되었습니다.")
    exit()

print(f"✓ LED OFF: {os.path.basename(path_off)}")

# ==========================================
# 2. 이미지 로드 및 차분 계산
# ==========================================
print("\n이미지 로드 중...")
img_on = load_image_with_hangul(path_on)
img_off = load_image_with_hangul(path_off)

if img_on is None or img_off is None:
    print("❌ 이미지를 읽을 수 없습니다.")
    exit()

# 차분 이미지 계산
diff_img = cv2.absdiff(img_on, img_off)
print("✓ 차분 이미지 계산 완료")

# 리사이징 (화면에 맞게)
height, width = diff_img.shape[:2]
scale_ratio = 800 / width
new_dim = (800, int(height * scale_ratio))
diff_img_resized = cv2.resize(diff_img, new_dim)

print(f"✓ 이미지 크기: {diff_img_resized.shape[1]}x{diff_img_resized.shape[0]}")

# ==========================================
# 3. HSV 필터링 GUI
# ==========================================
print("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
print("📌 단축키:")
print("   R 키: 이미지 다시 선택")
print("   Q 키: 종료")
print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n")

# 윈도우 및 트랙바 생성
cv2.namedWindow('Diff HSV Tuner')

cv2.createTrackbar('H Min', 'Diff HSV Tuner', 0, 179, nothing)
cv2.createTrackbar('H Max', 'Diff HSV Tuner', 179, 179, nothing)
cv2.createTrackbar('S Min', 'Diff HSV Tuner', 0, 255, nothing)
cv2.createTrackbar('S Max', 'Diff HSV Tuner', 255, 255, nothing)
cv2.createTrackbar('V Min', 'Diff HSV Tuner', 50, 255, nothing)
cv2.createTrackbar('V Max', 'Diff HSV Tuner', 255, 255, nothing)
cv2.createTrackbar('Threshold', 'Diff HSV Tuner', 50, 255, nothing)

while True:
    # 트랙바 값 읽기
    h_min = cv2.getTrackbarPos('H Min', 'Diff HSV Tuner')
    h_max = cv2.getTrackbarPos('H Max', 'Diff HSV Tuner')
    s_min = cv2.getTrackbarPos('S Min', 'Diff HSV Tuner')
    s_max = cv2.getTrackbarPos('S Max', 'Diff HSV Tuner')
    v_min = cv2.getTrackbarPos('V Min', 'Diff HSV Tuner')
    v_max = cv2.getTrackbarPos('V Max', 'Diff HSV Tuner')
    threshold = cv2.getTrackbarPos('Threshold', 'Diff HSV Tuner')

    # 임계값 적용 (밝은 차이만 남기기)
    diff_gray = cv2.cvtColor(diff_img_resized, cv2.COLOR_BGR2GRAY)
    _, thresh_mask = cv2.threshold(diff_gray, threshold, 255, cv2.THRESH_BINARY)
    
    # 임계값 마스크 적용한 차분 이미지
    diff_filtered = cv2.bitwise_and(diff_img_resized, diff_img_resized, mask=thresh_mask)

    # HSV 변환 및 마스킹
    hsv = cv2.cvtColor(diff_filtered, cv2.COLOR_BGR2HSV)
    lower_bound = np.array([h_min, s_min, v_min])
    upper_bound = np.array([h_max, s_max, v_max])
    hsv_mask = cv2.inRange(hsv, lower_bound, upper_bound)

    # 노이즈 제거
    kernel = np.ones((3,3), np.uint8)
    hsv_mask = cv2.erode(hsv_mask, kernel, iterations=1)
    hsv_mask = cv2.dilate(hsv_mask, kernel, iterations=2)

    # 결과 이미지
    result_img = diff_img_resized.copy()
    
    # 중심점 찾기
    contours, _ = cv2.findContours(hsv_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if contours:
        # 가장 큰 컨투어
        max_contour = max(contours, key=cv2.contourArea)
        
        if cv2.contourArea(max_contour) > 10:
                # 중심점 표시 제거됨 (사용자 요청)
                
                # 컨투어(윤곽선)만 표시
                cv2.drawContours(result_img, [max_contour], -1, (255, 0, 255), 2)

    # 화면 출력 (가로 구성: 왼쪽 결과, 오른쪽 마스크)
    hsv_mask_bgr = cv2.cvtColor(hsv_mask, cv2.COLOR_GRAY2BGR)
    
    # 가로로 쌓기
    stacked = np.hstack((result_img, hsv_mask_bgr))
    
    # 안내 텍스트
    cv2.putText(stacked, "Result (Diff + Markers)", (10, 25), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
    cv2.putText(stacked, "HSV Mask", (result_img.shape[1] + 10, 25), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
    
    # 단축키 안내
    cv2.putText(stacked, "Press 'R' to Reload | 'Q' to Quit", 
               (10, stacked.shape[0] - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    
    cv2.imshow('Diff HSV Tuner', stacked)

    # 키 입력 처리
    key = cv2.waitKey(1) & 0xFF
    
    if key == ord('q') or key == ord('Q'):
        print("\n프로그램을 종료합니다.")
        break
    
    elif key == ord('r') or key == ord('R'):
        # 이미지 다시 선택
        print("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
        print("새 이미지를 선택하세요...")
        
        new_path_on = select_image("1. LED ON 이미지 선택")
        if new_path_on:
            new_path_off = select_image("2. LED OFF 이미지 선택")
            
            if new_path_off:
                new_on = load_image_with_hangul(new_path_on)
                new_off = load_image_with_hangul(new_path_off)
                
                if new_on is not None and new_off is not None:
                    print(f"✓ LED ON: {os.path.basename(new_path_on)}")
                    print(f"✓ LED OFF: {os.path.basename(new_path_off)}")
                    
                    diff_img = cv2.absdiff(new_on, new_off)
                    diff_img_resized = cv2.resize(diff_img, new_dim)
                    print("✓ 새 차분 이미지 계산 완료\n")
                else:
                    print("❌ 이미지를 읽을 수 없습니다.\n")
            else:
                print("❌ LED OFF 선택 취소\n")
        else:
            print("❌ LED ON 선택 취소\n")

cv2.destroyAllWindows()
