import cv2
import numpy as np
from tkinter import Tk, filedialog
import os

def nothing(x):
    pass

def load_image_with_hangul(image_path):
    """한글 경로를 지원하는 이미지 로드 함수"""
    # 한글 경로 문제 해결: numpy로 먼저 읽고 디코딩
    with open(image_path, 'rb') as f:
        image_array = np.frombuffer(f.read(), dtype=np.uint8)
    img = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
    return img

def select_image():
    """파일 선택 다이얼로그"""
    root = Tk()
    root.withdraw()
    
    initial_dir = os.path.dirname(os.path.abspath(__file__))
    
    image_path = filedialog.askopenfilename(
        initialdir=initial_dir,
        title="레이저 이미지 선택",
        filetypes=(
            ("이미지 파일", "*.jpg *.jpeg *.png *.bmp"),
            ("모든 파일", "*.*")
        )
    )
    
    root.destroy()
    return image_path

# === 초기 이미지 선택 ===
print("이미지 파일을 선택하세요...")
image_path = select_image()

if not image_path:
    print("파일이 선택되지 않았습니다. 종료합니다.")
    exit()

print(f"선택된 파일: {image_path}")

# 이미지 불러오기 (한글 경로 지원)
original_img = load_image_with_hangul(image_path)

if original_img is None:
    print(f"이미지를 읽을 수 없습니다: {image_path}")
    exit()

# 이미지 리사이징
height, width = original_img.shape[:2]
scale_ratio = 800 / width
new_dim = (800, int(height * scale_ratio))
img = cv2.resize(original_img, new_dim)

# 윈도우 생성
cv2.namedWindow('Laser Tuner')

# 트랙바 생성
cv2.createTrackbar('H Min', 'Laser Tuner', 120, 179, nothing)
cv2.createTrackbar('H Max', 'Laser Tuner', 170, 179, nothing)
cv2.createTrackbar('S Min', 'Laser Tuner', 50, 255, nothing)
cv2.createTrackbar('S Max', 'Laser Tuner', 255, 255, nothing)
cv2.createTrackbar('V Min', 'Laser Tuner', 200, 255, nothing)
cv2.createTrackbar('V Max', 'Laser Tuner', 255, 255, nothing)

print("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
print("📌 단축키 안내:")
print("   L 키: 새 이미지 로드")
print("   Q 키: 종료")
print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n")

while True:
    # 트랙바 값 읽기
    h_min = cv2.getTrackbarPos('H Min', 'Laser Tuner')
    h_max = cv2.getTrackbarPos('H Max', 'Laser Tuner')
    s_min = cv2.getTrackbarPos('S Min', 'Laser Tuner')
    s_max = cv2.getTrackbarPos('S Max', 'Laser Tuner')
    v_min = cv2.getTrackbarPos('V Min', 'Laser Tuner')
    v_max = cv2.getTrackbarPos('V Max', 'Laser Tuner')

    # HSV 변환 및 마스킹
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lower_bound = np.array([h_min, s_min, v_min])
    upper_bound = np.array([h_max, s_max, v_max])
    mask = cv2.inRange(hsv, lower_bound, upper_bound)

    # 노이즈 제거
    kernel = np.ones((3,3), np.uint8)
    mask = cv2.erode(mask, kernel, iterations=1)
    mask = cv2.dilate(mask, kernel, iterations=1)

    # 무게중심 찾기
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    result_img = img.copy()
    
    if contours:
        max_contour = max(contours, key=cv2.contourArea)
        
        if cv2.contourArea(max_contour) > 10:
            M = cv2.moments(max_contour)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])

                # 결과 표시
                cv2.circle(result_img, (cx, cy), 10, (0, 255, 0), 2)
                cv2.drawMarker(result_img, (cx, cy), (0, 0, 255), 
                              markerType=cv2.MARKER_CROSS, thickness=2)
                cv2.putText(result_img, f"Laser: ({cx}, {cy})", (cx - 50, cy - 20), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # 화면 출력
    mask_bgr = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    stacked = np.hstack((result_img, mask_bgr))
    
    # 상단에 단축키 안내 표시
    cv2.putText(stacked, "Press 'L' to Load new image | 'Q' to Quit", 
               (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
    
    cv2.imshow('Laser Tuner', stacked)

    # 키 입력 처리
    key = cv2.waitKey(1) & 0xFF
    
    if key == ord('q') or key == ord('Q'):
        # 종료
        print("프로그램을 종료합니다.")
        break
    
    elif key == ord('l') or key == ord('L'):
        # 새 이미지 로드
        print("\n새 이미지를 선택하세요...")
        new_path = select_image()
        
        if new_path:
            new_img = load_image_with_hangul(new_path)
            
            if new_img is not None:
                print(f"새 이미지 로드 완료: {new_path}")
                original_img = new_img
                
                # 리사이징
                height, width = original_img.shape[:2]
                scale_ratio = 800 / width
                new_dim = (800, int(height * scale_ratio))
                img = cv2.resize(original_img, new_dim)
            else:
                print("이미지를 읽을 수 없습니다.")
        else:
            print("파일 선택이 취소되었습니다.")

cv2.destroyAllWindows()