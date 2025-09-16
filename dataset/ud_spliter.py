#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
split_ud_nocli.py
- 파일명에 'ud' 표기가 있는 보정본과 원본 이미지를 자동 분리
- CLI 인자 없이 바로 실행 가능 (더블클릭 OK)
- (옵션) YOLO 라벨(.txt)도 함께 이동 가능
"""

from pathlib import Path
import shutil, re, sys
try:
    # GUI 폴더 선택(옵션)
    import tkinter as tk
    from tkinter import filedialog, messagebox
    TK_OK = True
except Exception:
    TK_OK = False

# ====================== CONFIG ======================
# [1] 분리할 이미지들이 들어있는 폴더 (없으면 GUI로 선택)
ROOT_DIR = r""  # 예: r"D:\retro\images"  (빈 문자열이면 실행 시 폴더 선택 창 표시)

# [2] 라벨 폴더 (YOLO .txt). 라벨이 아직 없다면 비워두세요(None).
LABELS_DIR = None  # 예: r"D:\retro\labels" 또는 None(같은 폴더에서 찾기)

# [3] 라벨도 함께 이동할지 여부 (아직 라벨링 전이면 False)
MOVE_LABELS = False

# [4] 하위 폴더까지 재귀적으로 찾기
RECURSIVE = False  # True면 하위 폴더 포함

# [5] UD 인식 모드
#  - "suffix": 파일명 끝이 *_ud, *-ud, *.ud 인 것만 UD로 간주(권장, 오탐 적음)
#  - "any": 파일명 어딘가에 'ud'가 포함되면 UD로 간주(유연하지만 'student' 같은 오탐 가능)
UD_MATCH_MODE = "suffix"  # "suffix" or "any"

# [6] 분리 결과 폴더 이름
DIRNAMES = {
    "img_ud":   "images_ud",
    "img_raw":  "images_raw",
    "lbl_ud":   "labels_ud",
    "lbl_raw":  "labels_raw",
}

# [7] 찾아볼 이미지 확장자
IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")

# [8] 드라이런(이동하지 않고 계획만 콘솔 출력)
DRY_RUN = False

# [9] 더블클릭 실행 시 창이 바로 닫히지 않게 마지막에 일시정지
PAUSE_ON_EXIT = True
# ====================================================


def choose_root_via_gui(title="분리할 이미지 폴더를 선택하세요"):
    if not TK_OK:
        return None
    root = tk.Tk(); root.withdraw()
    path = filedialog.askdirectory(title=title)
    root.destroy()
    return path or None

def build_ud_regex():
    if UD_MATCH_MODE.lower() == "any":
        # 파일명 어딘가에 'ud' (대소문자 무시)
        return re.compile(r"(?i)ud")
    # 기본: 접미사 패턴 (_ud, -ud, .ud 로 끝나는 경우)
    return re.compile(r"(?i)[_\-\.]ud$")

def is_ud_stem(stem: str, rx: re.Pattern):
    return rx.search(stem) is not None

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def unique_path(dst: Path) -> Path:
    """동일 파일명이 있으면 _1, _2 ...를 붙여 충돌 회피"""
    if not dst.exists():
        return dst
    base, suf = dst.stem, dst.suffix
    parent = dst.parent
    i = 1
    while True:
        cand = parent / f"{base}_{i}{suf}"
        if not cand.exists():
            return cand
        i += 1

def move_file(src: Path, dst_dir: Path):
    ensure_dir(dst_dir)
    dst = unique_path(dst_dir / src.name)
    if DRY_RUN:
        print(f"[DRY] {src} -> {dst}")
    else:
        shutil.move(str(src), str(dst))

def main():
    # 1) ROOT_DIR 확보
    root_dir = ROOT_DIR.strip()
    if not root_dir:
        if TK_OK:
            sel = choose_root_via_gui()
            if not sel:
                print("작업이 취소되었습니다.")
                return
            root_dir = sel
        else:
            # GUI 불가 환경: 입력으로 대체
            root_dir = input("[입력] 분리할 이미지 폴더 경로를 적어주세요: ").strip()
            if not root_dir:
                print("경로가 비어 있습니다. 종료합니다.")
                return

    ROOT = Path(root_dir).resolve()
    if not ROOT.exists():
        print(f"[에러] 폴더가 존재하지 않습니다: {ROOT}")
        return

    # 2) LABELS_DIR
    LABELS = Path(LABELS_DIR).resolve() if LABELS_DIR else None
    if MOVE_LABELS and LABELS_DIR and not Path(LABELS_DIR).exists():
        print(f"[경고] LABELS_DIR가 존재하지 않습니다. 라벨 이동을 건너뜁니다: {LABELS_DIR}")
        labels_move = False
    else:
        labels_move = MOVE_LABELS

    # 3) 대상 파일 수집
    if RECURSIVE:
        imgs = [p for p in ROOT.rglob("*") if p.is_file() and p.suffix.lower() in IMG_EXTS]
    else:
        imgs = [p for p in ROOT.iterdir() if p.is_file() and p.suffix.lower() in IMG_EXTS]

    print(f"[INFO] 이미지 {len(imgs)}장 발견 @ {ROOT}")

    # 4) 정규식 준비
    rx = build_ud_regex()

    # 5) 목적지 폴더 준비
    IMG_UD = ROOT / DIRNAMES["img_ud"]
    IMG_RAW = ROOT / DIRNAMES["img_raw"]
    if labels_move:
        if LABELS:
            LBL_UD  = Path(LABELS) / DIRNAMES["lbl_ud"]
            LBL_RAW = Path(LABELS) / DIRNAMES["lbl_raw"]
        else:
            LBL_UD  = ROOT / DIRNAMES["lbl_ud"]
            LBL_RAW = ROOT / DIRNAMES["lbl_raw"]
    else:
        LBL_UD = LBL_RAW = None  # 사용 안 함

    # 6) 분리 루프
    c_img_ud = c_img_raw = c_lbl_ud = c_lbl_raw = c_lbl_miss = 0

    for img in imgs:
        stem = img.stem.lower()
        is_ud = is_ud_stem(stem, rx)

        # 이미지 이동
        if is_ud:
            move_file(img, IMG_UD); c_img_ud += 1
        else:
            move_file(img, IMG_RAW); c_img_raw += 1

        # (옵션) 라벨 이동
        if labels_move:
            if LABELS:
                lbl = Path(LABELS) / f"{img.stem}.txt"
            else:
                lbl = img.with_suffix(".txt")
            if lbl.exists():
                if is_ud:
                    move_file(lbl, LBL_UD); c_lbl_ud += 1
                else:
                    move_file(lbl, LBL_RAW); c_lbl_raw += 1
            else:
                c_lbl_miss += 1

    # 7) 결과 출력
    print("\n[RESULT]")
    print(f"  UD 이미지 이동:   {c_img_ud}")
    print(f"  RAW 이미지 이동:  {c_img_raw}")
    if labels_move:
        print(f"  UD 라벨 이동:     {c_lbl_ud}")
        print(f"  RAW 라벨 이동:    {c_lbl_raw}")
        print(f"  라벨 미발견:      {c_lbl_miss}")
    if DRY_RUN:
        print("  (DRY-RUN: 실제 이동 없음)")

if __name__ == "__main__":
    try:
        main()
    finally:
        if PAUSE_ON_EXIT:
            try:
                # 더블클릭 실행 시 창이 바로 닫히지 않도록
                input("\n완료되었습니다. Enter 키를 누르면 창을 닫습니다...")
            except Exception:
                pass
