import cv2
import numpy as np
import matplotlib.pyplot as plt

def show_laser_coordinates(image_path_1, image_path_2, roi_size=800):
    # --- 1. 이미지 읽기 헬퍼 함수 ---
    def imread_korean(path):
        try:
            with open(path, 'rb') as f:
                img_array = np.frombuffer(f.read(), dtype=np.uint8)
                img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
            return img
        except Exception as e:
            return None
    
    img1_full = imread_korean(image_path_1)
    img2_full = imread_korean(image_path_2)

    if img1_full is None or img2_full is None:
        print("❌ 이미지를 찾을 수 없습니다.")
        return

    if img1_full.shape != img2_full.shape:
        img2_full = cv2.resize(img2_full, (img1_full.shape[1], img1_full.shape[0]))

    # --- 2. ROI 설정 (중앙 자르기) ---
    h, w = img1_full.shape[:2]
    center_x, center_y = w // 2, h // 2
    
    half_roi = roi_size // 2
    x1 = max(0, center_x - half_roi) # ROI 시작 X (Offset X)
    y1 = max(0, center_y - half_roi) # ROI 시작 Y (Offset Y)
    x2 = min(w, center_x + half_roi)
    y2 = min(h, center_y + half_roi)

    # 원본 보존을 위해 복사본 생성 (그림 그리기용)
    img_full_vis = img2_full.copy() # 전체 화면에 그릴 것
    
    # ROI 자르기
    img1_roi = img1_full[y1:y2, x1:x2]
    img2_roi = img2_full[y1:y2, x1:x2]
    
    # ROI 시각화용 복사본
    img_roi_vis = img2_roi.copy()

    # --- 3. 이미지 처리 (Diff & Threshold) ---
    gray1 = cv2.cvtColor(img1_roi, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2_roi, cv2.COLOR_BGR2GRAY)
    gray1 = cv2.GaussianBlur(gray1, (5, 5), 0)
    gray2 = cv2.GaussianBlur(gray2, (5, 5), 0)
    
    diff_roi = cv2.absdiff(gray1, gray2)
    _, binary_diff_roi = cv2.threshold(diff_roi, 30, 255, cv2.THRESH_BINARY)

    # --- 4. 레이저 중심 좌표 찾기 (Moments) ---
    # 흰색 덩어리(Contour)를 찾습니다.
    contours, _ = cv2.findContours(binary_diff_roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    laser_detected = False
    cx, cy = 0, 0 # ROI 내 좌표
    gx, gy = 0, 0 # 전체 좌표 (Global)

    if contours:
        # 가장 큰 덩어리를 레이저로 간주 (노이즈 제거 효과)
        largest_contour = max(contours, key=cv2.contourArea)
        
        # 모멘트 계산
        M = cv2.moments(largest_contour)
        
        # 무게 중심 계산 (분모가 0이 아닐 때만)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            
            # 📍 좌표 변환: ROI 좌표 -> 전체 좌표
            gx = cx + x1
            gy = cy + y1
            
            laser_detected = True

    # --- 5. 시각화 (좌표 찍기) ---
    if laser_detected:
        # (A) ROI 이미지에 십자가 그리기 (빨간색)
        cv2.drawMarker(img_roi_vis, (cx, cy), (0, 0, 255), markerType=cv2.MARKER_CROSS, markerSize=20, thickness=2)
        cv2.putText(img_roi_vis, f"ROI: {cx},{cy}", (cx + 10, cy - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        # (B) 전체 이미지에 십자가 그리기 (빨간색) + ROI 박스 (초록색)
        cv2.rectangle(img_full_vis, (x1, y1), (x2, y2), (0, 255, 0), 5) # ROI 박스
        cv2.drawMarker(img_full_vis, (gx, gy), (0, 0, 255), markerType=cv2.MARKER_CROSS, markerSize=50, thickness=5)
        cv2.putText(img_full_vis, f"Global: {gx},{gy}", (gx + 20, gy - 20), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)
        
        print(f"✅ 레이저 검출 성공!")
        print(f"   - ROI 내 좌표: ({cx}, {cy})")
        print(f"   - 전체 좌표  : ({gx}, {gy})")
    else:
        print("⚠️ 레이저를 찾지 못했습니다.")

    # --- 6. Matplotlib 출력 ---
    plt.figure(figsize=(14, 10))

    # 1. ROI 결과 (Binary Mask)
    plt.subplot(2, 2, 1)
    plt.title("1. ROI Binary Mask (Processing)")
    plt.imshow(binary_diff_roi, cmap='gray')
    plt.axis('off')

    # 2. ROI 결과 (좌표 표시)
    plt.subplot(2, 2, 2)
    plt.title(f"2. Detected Laser in ROI ({cx}, {cy})")
    plt.imshow(cv2.cvtColor(img_roi_vis, cv2.COLOR_BGR2RGB))
    plt.axis('off')

    # 3. 전체 화면 결과 (좌표 표시)
    plt.subplot(2, 1, 2) # 아래쪽 전체 사용
    plt.title(f"3. Full Image with Coordinates ({gx}, {gy})")
    plt.imshow(cv2.cvtColor(img_full_vis, cv2.COLOR_BGR2RGB))
    plt.axis('off')

    plt.tight_layout()
    plt.show()

# --- 실행 ---
show_laser_coordinates(
    'C:/Users/gmlwn/OneDrive/바탕 화면/레이저필터데이터셋/captures_gui_20251126_203956/snap_20251126_204715_ud.jpg',
    'C:/Users/gmlwn/OneDrive/바탕 화면/레이저필터데이터셋/captures_gui_20251126_203956/snap_20251126_204724_ud.jpg',
    roi_size=200
)