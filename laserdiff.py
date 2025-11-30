import cv2
import numpy as np
import matplotlib.pyplot as plt

def show_center_diff_image(image_path_1, image_path_2, roi_size=800):
    # 1. 이미지 읽어오기 (한글 경로 지원 함수)
    def imread_korean(path):
        try:
            with open(path, 'rb') as f:
                img_array = np.frombuffer(f.read(), dtype=np.uint8)
                img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
            return img
        except Exception as e:
            print(f"❌ 이미지 읽기 실패: {path}")
            print(f"   에러: {e}")
            return None
    
    img1_full = imread_korean(image_path_1)
    img2_full = imread_korean(image_path_2)

    # 이미지가 제대로 읽혔는지 확인
    if img1_full is None or img2_full is None:
        print("❌ 이미지를 찾을 수 없습니다. 경로를 확인해주세요.")
        return

    # 두 이미지 크기가 다르면 계산이 안되므로 리사이즈
    if img1_full.shape != img2_full.shape:
        img2_full = cv2.resize(img2_full, (img1_full.shape[1], img1_full.shape[0]))

    # ==========================================
    # 📍 ROI (중앙 자르기) 로직 추가
    # ==========================================
    h, w = img1_full.shape[:2] # 전체 높이, 너비
    center_x, center_y = w // 2, h // 2 # 중앙 좌표
    
    # ROI 시작/끝 좌표 계산 (좌상단, 우하단)
    half_roi = roi_size // 2
    x1 = max(0, center_x - half_roi)
    y1 = max(0, center_y - half_roi)
    x2 = min(w, center_x + half_roi)
    y2 = min(h, center_y + half_roi)

    # 이미지 자르기 (Slicing) -> 이제부터 이 변수들로 연산함
    img1_roi = img1_full[y1:y2, x1:x2]
    img2_roi = img2_full[y1:y2, x1:x2]
    
    print(f"ℹ️ 전체 해상도: {w}x{h}")
    print(f"ℹ️ ROI 적용됨: 중앙을 기준으로 {roi_size}x{roi_size} 크기로 자름")

    # ==========================================
    # ⚙️ 이미지 처리 (ROI 이미지에만 적용)
    # ==========================================
    
    # 2. 흑백 변환
    gray1 = cv2.cvtColor(img1_roi, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2_roi, cv2.COLOR_BGR2GRAY)

    # 3. 노이즈 제거 (Gaussian Blur)
    gray1 = cv2.GaussianBlur(gray1, (5, 5), 0)
    gray2 = cv2.GaussianBlur(gray2, (5, 5), 0)

    # 4. 차분 이미지 계산 (Absolute Difference)
    diff_roi = cv2.absdiff(gray1, gray2)

    # 5. 이진화 (Thresholding)
    # 레이저가 밝다면 30~50 정도, 흐릿하면 20 정도로 조절
    _, binary_diff_roi = cv2.threshold(diff_roi, 30, 255, cv2.THRESH_BINARY)

    # ==========================================
    # 📊 시각화 (matplotlib)
    # ==========================================
    
    # 시각화를 위해 원본(Full) 이미지에 초록색 네모 박스 그리기 (어디 잘랐는지 확인용)
    img1_vis = img1_full.copy()
    img2_vis = img2_full.copy()
    cv2.rectangle(img1_vis, (x1, y1), (x2, y2), (0, 255, 0), 10) # 두께 10
    cv2.rectangle(img2_vis, (x1, y1), (x2, y2), (0, 255, 0), 10)

    plt.figure(figsize=(12, 10))

    # 1. 원본 1 (전체 + ROI 박스)
    plt.subplot(2, 2, 1)
    plt.title("Full Image 1 (Green Box = ROI)")
    plt.imshow(cv2.cvtColor(img1_vis, cv2.COLOR_BGR2RGB))
    plt.axis('off')

    # 2. 원본 2 (전체 + ROI 박스)
    plt.subplot(2, 2, 2)
    plt.title("Full Image 2 (Green Box = ROI)")
    plt.imshow(cv2.cvtColor(img2_vis, cv2.COLOR_BGR2RGB))
    plt.axis('off')

    # 3. ROI 영역의 차이 (Gray) - 확대된 모습
    plt.subplot(2, 2, 3)
    plt.title(f"ROI Difference Map ({roi_size}x{roi_size})")
    plt.imshow(diff_roi, cmap='gray')
    plt.axis('off')

    # 4. ROI 영역의 레이저 검출 (Binary) - 확대된 모습
    plt.subplot(2, 2, 4)
    plt.title("Detected Laser in ROI")
    plt.imshow(binary_diff_roi, cmap='gray')
    plt.axis('off')

    plt.tight_layout()
    plt.show()

# --- 실행 부분 ---
# 800x800 크기로 중앙만 잘라서 비교합니다.
show_center_diff_image(
    'C:/Users/gmlwn/OneDrive/바탕 화면/레이저필터데이터셋/captures_gui_20251126_203956/snap_20251126_204715_ud.jpg',
    'C:/Users/gmlwn/OneDrive/바탕 화면/레이저필터데이터셋/captures_gui_20251126_203956/snap_20251126_204724_ud.jpg',
    roi_size=800
)