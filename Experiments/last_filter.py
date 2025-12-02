#!/usr/bin/env python3
"""
LED 차분 이미지 자동 필터 선택기 (Auto Filter Selector)
- 1차 시도 (Universe Filter): 가까운 거리(노란색 허용) 우선 탐색
- 2차 시도 (RGB Two Filter): 실패 시 먼 거리(엄격한 빨강) 재탐색
- [기능 추가] 'R' 키를 눌러 이미지 재선택 가능
"""

import cv2
import numpy as np
from tkinter import Tk, filedialog
import os
import sys

# ==========================================
# 사용자 요청 파라미터 설정 (하드코딩)
# ==========================================

# [Filter 1] Universe (가까운 거리/노란색 허용)
F1_BOOST = 0
F1_MIN_BRIGHT = 30
F1_WHITE_CUT = 70   # R > B
F1_YELLOW_RANGE = 60 # R - G > -60

# [Filter 2] RGB Two (먼 거리/엄격한 빨강)
F2_BOOST = 1
F2_MIN_RED = 40     # Min Brightness
F2_DIFF_G = 40      # R > G
F2_DIFF_B = 30      # R > B

# ==========================================
# 유틸리티 함수
# ==========================================
def load_image_with_hangul(image_path):
    try:
        with open(image_path, 'rb') as f:
            image_array = np.frombuffer(f.read(), dtype=np.uint8)
        img = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
        return img
    except Exception as e:
        print(f"❌ 이미지 로드 실패: {e}")
        return None

def select_two_images():
    print("\n[시스템] 파일 선택창을 엽니다...")
    root = Tk()
    root.withdraw()
    root.attributes('-topmost', True)

    initial_dir = os.path.dirname(os.path.abspath(__file__))
    
    print(">> 1. LED ON (빛 받은) 이미지를 선택하세요.")
    path_on = filedialog.askopenfilename(
        initialdir=initial_dir, title="1. LED ON 이미지 선택",
        filetypes=(("이미지 파일", "*.jpg *.jpeg *.png *.bmp"), ("모든 파일", "*.*")),
        parent=root
    )
    if not path_on: root.destroy(); return None, None
    
    print(">> 2. LED OFF (빛 없는) 이미지를 선택하세요.")
    path_off = filedialog.askopenfilename(
        initialdir=initial_dir, title="2. LED OFF 이미지 선택",
        filetypes=(("이미지 파일", "*.jpg *.jpeg *.png *.bmp"), ("모든 파일", "*.*")),
        parent=root
    )
    if not path_off: root.destroy(); return None, None

    root.destroy()
    return path_on, path_off

def preprocess_image(img, boost_on):
    """부스트 여부에 따른 전처리"""
    processed = img.copy()
    if boost_on == 1:
        processed = cv2.normalize(processed, None, 0, 255, cv2.NORM_MINMAX)
        processed = cv2.GaussianBlur(processed, (3, 3), 0)
    return processed

def apply_morphology(mask):
    """노이즈 제거"""
    kernel = np.ones((3,3), np.uint8)
    mask = cv2.erode(mask, kernel, iterations=1)
    mask = cv2.dilate(mask, kernel, iterations=2)
    return mask

# ==========================================
# 필터 로직 함수
# ==========================================

def run_filter_1_universe(diff_img):
    """
    1차 필터: Universe (노란색 허용)
    파라미터: 1, 30, 70, 60
    """
    current_img = preprocess_image(diff_img, F1_BOOST)
    
    B, G, R = cv2.split(current_img)
    R_int = R.astype(np.int16)
    G_int = G.astype(np.int16)
    B_int = B.astype(np.int16)

    # 로직 적용
    mask_bright = (R > F1_MIN_BRIGHT)
    mask_white_cut = (R_int - B_int) > F1_WHITE_CUT
    mask_color_range = (R_int - G_int) > -F1_YELLOW_RANGE

    final_mask = mask_bright & mask_white_cut & mask_color_range
    final_mask = final_mask.astype(np.uint8) * 255
    final_mask = apply_morphology(final_mask)
    
    return final_mask, current_img

def run_filter_2_rgb_two(diff_img):
    """
    2차 필터: RGB Two (엄격한 빨강)
    파라미터: 0, 40, 30, 30
    """
    current_img = preprocess_image(diff_img, F2_BOOST)
    
    B, G, R = cv2.split(current_img)
    R_int = R.astype(np.int16)
    G_int = G.astype(np.int16)
    B_int = B.astype(np.int16)

    # 로직 적용 (Strict Red)
    mask_abs = (R > F2_MIN_RED)
    mask_rg = (R_int - G_int) > F2_DIFF_G
    mask_rb = (R_int - B_int) > F2_DIFF_B

    final_mask = mask_abs & mask_rg & mask_rb
    final_mask = final_mask.astype(np.uint8) * 255
    final_mask = apply_morphology(final_mask)

    return final_mask, current_img

# ==========================================
# 메인 실행부
# ==========================================
def main():
    while True: # [수정] 무한 루프 시작
        # 1. 이미지 로드
        path_on, path_off = select_two_images()
        
        # 취소 시 종료
        if not path_on or not path_off:
            print("❌ 파일 선택 취소. 프로그램을 종료합니다.")
            break

        img_on = load_image_with_hangul(path_on)
        img_off = load_image_with_hangul(path_off)

        if img_on is None or img_off is None:
            print("❌ 이미지 로드 실패. 다시 시도해주세요.")
            continue

        # 2. 차분 이미지 및 리사이징
        diff_img_original = cv2.absdiff(img_on, img_off)
        height, width = diff_img_original.shape[:2]
        scale_ratio = 800 / width
        new_dim = (800, int(height * scale_ratio))
        diff_img_resized = cv2.resize(diff_img_original, new_dim)

        # 3. 자동 필터 선택 로직 실행
        print("\n-------------------------------------------")
        print("🚀 [1차 시도] Universe Filter 실행 중...")
        mask, view_img = run_filter_1_universe(diff_img_resized)
        
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        active_filter_name = "Filter 1: Universe (Green)"
        box_color = (0, 255, 0) # Green
        found = False

        # 1차 결과 확인
        if contours:
            max_cnt = max(contours, key=cv2.contourArea)
            if cv2.contourArea(max_cnt) > 5:
                found = True
                print("✅ 1차 필터에서 타겟 발견!")
        
        # 1차 실패 시 2차 실행
        if not found:
            print("⚠️ 1차 실패 -> [2차 시도] RGB Two Filter 실행 중...")
            mask, view_img = run_filter_2_rgb_two(diff_img_resized)
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            active_filter_name = "Filter 2: RGB Two (Purple)"
            box_color = (255, 0, 255) # Purple
            
            if contours:
                max_cnt = max(contours, key=cv2.contourArea)
                if cv2.contourArea(max_cnt) > 5:
                    found = True
                    print("✅ 2차 필터에서 타겟 발견!")
                else:
                    print("❌ 2차 필터에서도 실패.")
            else:
                print("❌ 2차 필터에서도 실패.")

        # 4. 결과 시각화
        result_view = view_img.copy()
        
        # 상단 정보 표시
        cv2.putText(result_view, f"Active: {active_filter_name}", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, box_color, 2)
        
        # [수정] 안내 문구 추가
        cv2.putText(result_view, "Press 'R' to Reload, 'Q' to Quit", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

        if found and contours:
            max_cnt = max(contours, key=cv2.contourArea)
            if cv2.contourArea(max_cnt) > 5:
                x, y, w, h = cv2.boundingRect(max_cnt)
                cv2.rectangle(result_view, (x, y), (x+w, y+h), box_color, 2)
                
                cx, cy = x + w//2, y + h//2
                cv2.drawMarker(result_view, (cx, cy), (0, 255, 255), cv2.MARKER_CROSS, 20, 2)
                
                info = f"Center:({cx},{cy}) Area:{int(cv2.contourArea(max_cnt))}"
                cv2.putText(result_view, info, (x, y-10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, box_color, 2)

        mask_bgr = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        stacked = np.hstack((result_view, mask_bgr))

        cv2.imshow('Auto Filter Selector', stacked)
        print("-------------------------------------------")
        print("⌨️  키보드 조작:")
        print("   [R] : 이미지 다시 선택")
        print("   [Q] : 프로그램 종료")
        print("-------------------------------------------\n")
        
        # [수정] 키 입력 대기 루프
        key = cv2.waitKey(0) & 0xFF
        if key == ord('r') or key == ord('R'):
            print("🔄 이미지를 다시 선택합니다...")
            cv2.destroyAllWindows()
            continue # 루프 처음으로
        elif key == ord('q') or key == ord('Q'):
            print("👋 프로그램을 종료합니다.")
            cv2.destroyAllWindows()
            break # 루프 종료

if __name__ == "__main__":
    main()