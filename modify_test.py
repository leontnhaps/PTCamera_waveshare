#!/usr/bin/env python3
"""
Com/test.py 수정 스크립트
Com_main.py 기반으로 LED ON/OFF 차분 방식으로 변경
"""

# Com/test.py 읽기
with open("Com/test.py", "r", encoding="utf-8") as f:
    content = f.read()

# 1. 스캔 중 실시간 YOLO 처리 부분 제거 (902-946줄 부분)
# "# === [핵심] 스캔 중 CSV에 YOLO 결과 기록 ===" 부터
# "except Exception as e:\n                                print(\"[SCAN][CSV] write err:\", e)" 까지 제거

old_realtime_yolo = '''                    # === [핵심] 스캔 중 CSV에 YOLO 결과 기록 ===
                    if self._scan_csv_writer is not None:
                            try:
                                # 파일명에서 pan/tilt 추출
                                m = self._fname_re.search(name)
                                pan_deg = float(m.group("pan")) if m else None
                                tilt_deg = float(m.group("tilt")) if m else None

                                # 원본 디코드
                                arr = np.frombuffer(data, np.uint8)
                                bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                                if bgr is None:
                                    raise RuntimeError("cv2.imdecode 실패")

                                # ★★★ 스캔 CSV/YOLO는 '항상 보정' ★★★
                                if self._ud_K is None or self._ud_D is None:
                                    # 안전장치: 혹시 start_scan 체크를 우회했을 때 대비
                                    ui_q.put(("toast", "❌ 보정 파라미터 없음 → 스캔 감지/기록 중단"))
                                    return
                                bgr = self._undistort_bgr(bgr)   # ← 여기서 반드시 언디스토트
                                H, W = bgr.shape[:2]             # ← 보정 이후의 W,H로 교체

                                # YOLO 보장 & 추론
                                if self._ensure_yolo_model_for_scan():
                                    res = self._yolo_model.predict(
                                        bgr,
                                        imgsz=self._scan_yolo_imgsz,
                                        conf=self._scan_yolo_conf,
                                        iou=float(self.yolo_iou.get()),
                                        device=self._yolo_device,
                                        verbose=False
                                    )[0]

                                    if res is not None and res.boxes is not None and len(res.boxes) > 0:
                                        for b in res.boxes:
                                            conf = float(b.conf.cpu().item() or 0.0)
                                            if conf < self._scan_yolo_conf: 
                                                continue
                                            cls = int(b.cls.cpu().item() or -1)
                                            x1,y1,x2,y2 = b.xyxy[0].cpu().numpy().tolist()
                                            cx = 0.5*(x1+x2); cy = 0.5*(y1+y2)
                                            # ★ CSV는 '보정 좌표계'로 기록됨
                                            self._scan_csv_writer.writerow([name, pan_deg, tilt_deg, cx, cy, x2-x1, y2-y1, conf, cls, W, H])
                            except Exception as e:
                                print("[SCAN][CSV] write err:", e)'''

new_comment = '''                    # === 스캔 중에는 이미지만 저장, YOLO는 스캔 완료 후 차분 처리 ==='''

content = content.replace(old_realtime_yolo, new_comment)

# 2. 스캔 완료 후 차분 + YOLO 처리 추가
# "elif et == \"done\":" 부분 찾아서 수정

old_done_section = '''                    elif et == "done":
                        # === CSV 닫기 ===
                        if self._scan_csv_file:
                            try:
                                self._scan_csv_file.flush()
                                self._scan_csv_file.close()
                            except Exception:
                                pass
                            finally:
                                self._scan_csv_file = None
                                self._scan_csv_writer = None
                                ui_q.put(("toast", f"CSV 저장 완료: {self._scan_csv_path}"))
                        if self.preview_enable.get():
                            self.toggle_preview()'''

new_done_section = '''                    elif et == "done":
                        # === 스캔 완료 후 LED ON/OFF 차분 이미지 처리 + YOLO + CSV 저장 ===
                        ui_q.put(("toast", "[SCAN] 스캔 완료! LED ON/OFF 차분 이미지 처리 시작..."))
                        
                        # 백그라운드 스레드에서 처리
                        def process_diff_and_yolo():
                            try:
                                import glob
                                from collections import defaultdict
                                
                                # 1. LED ON/OFF 이미지 그룹화
                                led_on_files = sorted(glob.glob(str(DEFAULT_OUT_DIR / "*_led_on.jpg")))
                                led_off_files = sorted(glob.glob(str(DEFAULT_OUT_DIR / "*_led_off.jpg")))
                                
                                # 파일명에서 pan/tilt 추출하여 매칭
                                pairs = defaultdict(dict)
                                fname_re = re.compile(r"img_t(?P<tilt>[+\\-]\\d{2,3})_p(?P<pan>[+\\-]\\d{2,3})_.*_led_(?P<state>on|off)\\.jpg$", re.IGNORECASE)
                                
                                for fpath in led_on_files + led_off_files:
                                    fname = os.path.basename(fpath)
                                    m = fname_re.search(fname)
                                    if m:
                                        pan = int(m.group("pan"))
                                        tilt = int(m.group("tilt"))
                                        state = m.group("state")
                                        key = (pan, tilt)
                                        pairs[key][state] = fpath
                                
                                ui_q.put(("toast", f"[DIFF] {len(pairs)}개 위치의 LED ON/OFF 쌍 발견"))
                                
                                # 2. CSV 파일 생성
                                sess = evt.get("session") or datetime.now().strftime("scan_%Y%m%d_%H%M%S")
                                csv_path = DEFAULT_OUT_DIR / f"{sess}_detections.csv"
                                
                                with open(csv_path, "w", newline="", encoding="utf-8") as csvfile:
                                    writer = csv.writer(csvfile)
                                    writer.writerow(["pan_deg", "tilt_deg", "cx", "cy", "w", "h", "conf", "cls", "W", "H"])
                                    
                                    # 3. YOLO 모델 로드 (GPU 사용)
                                    if not _YOLO_OK:
                                        ui_q.put(("toast", "❌ YOLO가 설치되지 않았습니다"))
                                        return
                                    
                                    yolo_wpath = self.yolo_wpath.get().strip()
                                    if not yolo_wpath:
                                        ui_q.put(("toast", "⚠️ YOLO 가중치 없음 - CSV는 빈 파일로 저장됩니다"))
                                        return
                                    
                                    yolo_model = YOLO(yolo_wpath)
                                    device = "cuda" if (torch and torch.cuda.is_available()) else "cpu"
                                    ui_q.put(("toast", f"[YOLO] Using device: {device}"))
                                    
                                    # 4. 각 위치별로 차분 이미지 생성 + YOLO 실행
                                    total_pairs = len(pairs)
                                    processed = 0
                                    detected_count = 0
                                    
                                    for (pan, tilt), files in sorted(pairs.items()):
                                        if "on" not in files or "off" not in files:
                                            continue  # ON/OFF 쌍이 없으면 스킵
                                        
                                        # LED ON/OFF 이미지 로드
                                        img_on = cv2.imread(files["on"])
                                        img_off = cv2.imread(files["off"])
                                        
                                        if img_on is None or img_off is None:
                                            continue
                                        
                                        # ★★★ Calibration 적용 (반드시!) ★★★
                                        if self._ud_K is not None and self._ud_D is not None:
                                            img_on = self._undistort_bgr(img_on)
                                            img_off = self._undistort_bgr(img_off)
                                        else:
                                            ui_q.put(("toast", "⚠️ Calibration 파라미터 없음 - 원본 이미지 사용"))
                                        
                                        # 차분 이미지 계산 (절대값)
                                        diff = cv2.absdiff(img_on, img_off)
                                        
                                        H, W = diff.shape[:2]
                                        
                                        # YOLO 실행 (차분 이미지에 대해)
                                        results = yolo_model.predict(
                                            diff,
                                            imgsz=self._scan_yolo_imgsz,
                                            conf=self._scan_yolo_conf,
                                            iou=float(self.yolo_iou.get()),
                                            device=device,
                                            verbose=False
                                        )[0]
                                        
                                        # 검출 결과 CSV에 저장
                                        if results.boxes is not None and len(results.boxes) > 0:
                                            for b in results.boxes:
                                                conf = float(b.conf.cpu().item() or 0.0)
                                                if conf < self._scan_yolo_conf:
                                                    continue
                                                cls = int(b.cls.cpu().item() or -1)
                                                x1, y1, x2, y2 = b.xyxy[0].cpu().numpy().tolist()
                                                cx = 0.5 * (x1 + x2)
                                                cy = 0.5 * (y1 + y2)
                                                writer.writerow([pan, tilt, cx, cy, x2-x1, y2-y1, conf, cls, W, H])
                                                detected_count += 1
                                        
                                        processed += 1
                                        if processed % 10 == 0:
                                            ui_q.put(("toast", f"[DIFF+YOLO] 처리 중... {processed}/{total_pairs} (검출: {detected_count})"))
                                
                                ui_q.put(("toast", f"✅ CSV 저장 완료: {csv_path} (총 {detected_count}개 검출)"))
                                
                            except Exception as e:
                                ui_q.put(("toast", f"❌ 차분 처리 실패: {e}"))
                                import traceback
                                traceback.print_exc()
                        
                        # 백그라운드 스레드 시작
                        threading.Thread(target=process_diff_and_yolo, daemon=True).start()
                        
                        # 프리뷰 재개
                        if self.preview_enable.get():
                            self.toggle_preview()'''

content = content.replace(old_done_section, new_done_section)

# 3. start_scan에 led_settle 파라미터 추가 (이미 있는지 확인)
if '"led_settle"' not in content:
    old_start_scan = '''        self.ctrl.send({
            "cmd":"scan_run",
            "pan_min":self.pan_min.get(),"pan_max":self.pan_max.get(),"pan_step":self.pan_step.get(),
            "tilt_min":self.tilt_min.get(),"tilt_max":self.tilt_max.get(),"tilt_step":self.tilt_step.get(),
            "speed":self.speed.get(),"acc":float(self.acc.get()),"settle":float(self.settle.get()),
            "width":self.width.get(),"height":self.height.get(),"quality":self.quality.get(),
            "session":datetime.now().strftime("scan_%Y%m%d_%H%M%S"),
            "hard_stop":self.hard_stop.get()
        })'''
    
    new_start_scan = '''        self.ctrl.send({
            "cmd":"scan_run",
            "pan_min":self.pan_min.get(),"pan_max":self.pan_max.get(),"pan_step":self.pan_step.get(),
            "tilt_min":self.tilt_min.get(),"tilt_max":self.tilt_max.get(),"tilt_step":self.tilt_step.get(),
            "speed":self.speed.get(),"acc":float(self.acc.get()),"settle":float(self.settle.get()),
            "led_settle":float(self.led_settle.get()),
            "width":self.width.get(),"height":self.height.get(),"quality":self.quality.get(),
            "session":datetime.now().strftime("scan_%Y%m%d_%H%M%S"),
            "hard_stop":self.hard_stop.get()
        })'''
    
    content = content.replace(old_start_scan, new_start_scan)

# 수정된 내용 저장
with open("Com/test.py", "w", encoding="utf-8") as f:
    f.write(content)

print("✅ Com/test.py 수정 완료!")
print("\n주요 변경사항:")
print("1. 스캔 중 실시간 YOLO 처리 제거")
print("2. 스캔 완료 후 LED ON/OFF 차분 이미지 처리 + YOLO + CSV 저장 추가")
print("3. start_scan에 led_settle 파라미터 추가")
