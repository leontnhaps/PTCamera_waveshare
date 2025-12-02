import os
import cv2
import torch
import pathlib
import numpy as np
from ultralytics import YOLO
from tkinter import Tk, filedialog

def non_max_suppression(boxes, scores, iou_threshold):
    # OpenCV NMS 사용
    if len(boxes) == 0:
        return []
    indices = cv2.dnn.NMSBoxes(boxes, scores, score_threshold=0.0, nms_threshold=iou_threshold)
    if len(indices) > 0:
        return indices.flatten()
    return []

def predict_with_tiling(model, img, rows=2, cols=3, overlap=0.15, conf=0.25, iou=0.45, device='cuda'):
    """
    이미지를 타일로 쪼개서 예측 후 결과 병합
    rows, cols: 행/열 개수 (2x3 = 6등분)
    overlap: 타일 간 겹치는 비율 (0.15 = 15%)
    """
    H, W = img.shape[:2]
    
    # 타일 크기 계산 (겹침 포함)
    tile_h = int(H / rows)
    tile_w = int(W / cols)
    
    # 겹침 크기
    ov_h = int(tile_h * overlap)
    ov_w = int(tile_w * overlap)
    
    # 실제 타일 크기 (겹침 포함)
    step_h = tile_h - ov_h
    step_w = tile_w - ov_w
    
    # 타일 좌표 생성
    tiles = []
    for y in range(0, H, step_h):
        for x in range(0, W, step_w):
            # 타일 영역 계산
            y2 = min(y + tile_h, H)
            x2 = min(x + tile_w, W)
            # 마지막 타일 크기 조정
            y1 = max(0, y2 - tile_h)
            x1 = max(0, x2 - tile_w)
            tiles.append((x1, y1, x2, y2))
            
            if x2 >= W: break
        if y2 >= H: break

    # 모든 타일 추론
    all_boxes = []   # [x, y, w, h]
    all_scores = []
    all_classes = []
    
    print(f"  -> {len(tiles)}개 타일로 분할 분석 중...", end="", flush=True)
    
    for i, (tx1, ty1, tx2, ty2) in enumerate(tiles):
        # 타일 잘라내기
        tile_img = img[ty1:ty2, tx1:tx2]
        
        # YOLO 추론
        results = model.predict(tile_img, conf=conf, iou=iou, device=device, verbose=False)[0]
        
        if results.boxes:
            for box in results.boxes:
                # 로컬 좌표
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                c = float(box.conf.cpu().numpy().item())
                cls = int(box.cls.cpu().numpy().item())
                
                # 글로벌 좌표로 변환
                gx1 = x1 + tx1
                gy1 = y1 + ty1
                gx2 = x2 + tx1
                gy2 = y2 + ty1
                
                # NMS를 위해 [x, y, w, h] 형식으로 저장
                w = gx2 - gx1
                h = gy2 - gy1
                
                all_boxes.append([int(gx1), int(gy1), int(w), int(h)])
                all_scores.append(c)
                all_classes.append(cls)
        print(".", end="", flush=True)
    print(" 완료")

    # 전체 결과에 대해 NMS 수행 (중복 박스 제거)
    if not all_boxes:
        return [], [], []

    indices = non_max_suppression(all_boxes, all_scores, iou_threshold=0.3) # 겹침 제거 강하게
    
    final_boxes = []
    final_scores = []
    final_classes = []
    
    for idx in indices:
        final_boxes.append(all_boxes[idx])
        final_scores.append(all_scores[idx])
        final_classes.append(all_classes[idx])
        
    return final_boxes, final_scores, final_classes

def draw_results(img, boxes, scores, classes):
    """결과 그리기"""
    plot_img = img.copy()
    for i, (x, y, w, h) in enumerate(boxes):
        x1, y1 = int(x), int(y)
        x2, y2 = int(x+w), int(y+h)
        score = scores[i]
        cls = classes[i]
        
        # 박스 그리기
        cv2.rectangle(plot_img, (x1, y1), (x2, y2), (0, 0, 255), 2)
        # 라벨
        label = f"{score:.2f}"
        cv2.putText(plot_img, label, (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
    return plot_img

def run_yolo_on_folder():
    # 1. Tkinter 초기화
    root = Tk()
    root.withdraw()

    print("=== YOLO 타일링(분할) 분석 도구 ===")
    print("이미지를 6등분(2x3)하여 작은 객체를 정밀 탐지합니다.")

    # 2. YOLO 가중치 파일 선택
    # print("\n[1] YOLO 가중치 파일(.pt)을 선택하세요...")
    # model_path = filedialog.askopenfilename(
    #     title="YOLO 가중치 파일 선택",
    #     filetypes=[("YOLO weights", "*.pt"), ("All files", "*.*")]
    # )
    # if not model_path: return
    
    # 편의를 위해 경로 고정 (필요시 주석 해제하여 선택창 사용)
    model_path = r"C:\Users\gmlwn\OneDrive\바탕 화면\ICon1학년\OpticalWPT\PTCamera_waveshare\yolov11m_diff.pt"
    if not os.path.exists(model_path):
         model_path = filedialog.askopenfilename(title="YOLO 가중치 파일 선택")

    print(f"-> 모델: {model_path}")

    # 3. 테스트할 이미지 폴더 선택
    print("\n[2] 테스트할 이미지가 있는 폴더를 선택하세요...")
    target_dir = filedialog.askdirectory(title="이미지 폴더 선택")
    if not target_dir: return
    target_path = pathlib.Path(target_dir)

    # 4. 결과 저장 폴더
    save_dir = target_path / "yolo_tiled_results"
    save_dir.mkdir(exist_ok=True)

    # 5. 모델 로드
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n[3] 모델 로딩 중... (Device: {device})")
    model = YOLO(model_path)

    # 6. 이미지 파일 찾기
    exts = ["*.jpg", "*.jpeg", "*.png", "*.bmp"]
    files = []
    for ext in exts:
        files.extend(target_path.glob(ext))
        files.extend(target_path.glob(ext.upper()))
    files = sorted(list(set(files)))
    
    if not files:
        print("이미지가 없습니다.")
        return

    print(f"\n[4] 총 {len(files)}장 처리 시작 (2x3 분할)...")

    # 7. 실행
    count = 0
    for i, fpath in enumerate(files):
        try:
            img = cv2.imread(str(fpath))
            if img is None: continue

            # ★ 타일링 추론 실행 ★
            # rows=2, cols=3 -> 6등분
            # overlap=0.15 -> 15% 겹치게 잘라서 경계선 객체 보호
            # conf=0.20 -> 신뢰도 낮춰서 작은 것도 잡기
            boxes, scores, classes = predict_with_tiling(
                model, img, 
                rows=2, cols=3, 
                overlap=0.15, 
                conf=0.20, 
                iou=0.45, 
                device=device
            )

            # 결과 그리기
            res_img = draw_results(img, boxes, scores, classes)

            # 저장
            save_path = save_dir / f"res_{fpath.name}"
            cv2.imwrite(str(save_path), res_img)
            
            count += 1
            print(f"[{i+1}/{len(files)}] {fpath.name} -> {len(boxes)}개 검출")

        except Exception as e:
            print(f"에러 ({fpath.name}): {e}")
            import traceback
            traceback.print_exc()

    print(f"\n=== 완료! 저장 위치: {save_dir} ===")

if __name__ == "__main__":
    run_yolo_on_folder()
