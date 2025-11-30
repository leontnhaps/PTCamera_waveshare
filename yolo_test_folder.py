import os
import cv2
import torch
import pathlib
from ultralytics import YOLO
from tkinter import Tk, filedialog

def run_yolo_on_folder():
    # 1. Tkinter 초기화 (창 숨김)
    root = Tk()
    root.withdraw()

    print("=== YOLO 모델 성능 평가 도구 ===")

    # 2. YOLO 가중치 파일 선택
    print("\n[1] YOLO 가중치 파일(.pt)을 선택하세요...")
    model_path = filedialog.askopenfilename(
        title="YOLO 가중치 파일 선택",
        filetypes=[("YOLO weights", "*.pt"), ("All files", "*.*")]
    )
    if not model_path:
        print("취소되었습니다.")
        return
    print(f"-> 선택된 모델: {model_path}")

    # 3. 테스트할 이미지 폴더 선택
    print("\n[2] 테스트할 이미지가 있는 폴더를 선택하세요...")
    target_dir = filedialog.askdirectory(title="이미지 폴더 선택")
    if not target_dir:
        print("취소되었습니다.")
        return
    target_path = pathlib.Path(target_dir)
    print(f"-> 선택된 폴더: {target_path}")

    # 4. 결과 저장 폴더 생성
    save_dir = target_path / "yolo_results"
    save_dir.mkdir(exist_ok=True)
    print(f"-> 결과 저장 경로: {save_dir}")

    # 5. YOLO 모델 로드
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n[3] 모델 로딩 중... (Device: {device})")
    try:
        model = YOLO(model_path)
    except Exception as e:
        print(f"모델 로드 실패: {e}")
        return

    # 6. 이미지 파일 찾기
    exts = ["*.jpg", "*.jpeg", "*.png", "*.bmp"]
    files = []
    for ext in exts:
        files.extend(target_path.glob(ext))
        files.extend(target_path.glob(ext.upper()))
    
    files = sorted(list(set(files)))
    total = len(files)
    
    if total == 0:
        print("폴더에 이미지가 없습니다.")
        return

    print(f"\n[4] 총 {total}장의 이미지 처리를 시작합니다...")

    # 7. 추론 및 저장
    count = 0
    for i, fpath in enumerate(files):
        try:
            # 이미지 읽기
            img = cv2.imread(str(fpath))
            if img is None:
                print(f"이미지 읽기 실패: {fpath.name}")
                continue

            # YOLO 추론
            # conf: 신뢰도 임계값 (0.5), iou: NMS 임계값 (0.45)
            results = model.predict(
                img,
                conf=0.50,
                iou=0.45,
                device=device,
                verbose=False
            )[0]

            # 결과 시각화 (Bounding Box 그리기)
            res_plotted = results.plot()

            # 저장
            save_path = save_dir / f"res_{fpath.name}"
            cv2.imwrite(str(save_path), res_plotted)
            
            count += 1
            print(f"[{i+1}/{total}] 저장 완료: {save_path.name} ({len(results.boxes)}개 검출)")

        except Exception as e:
            print(f"에러 발생 ({fpath.name}): {e}")

    print(f"\n=== 완료! 총 {count}장의 결과 이미지가 저장되었습니다. ===")
    print(f"저장 위치: {save_dir}")

if __name__ == "__main__":
    run_yolo_on_folder()
