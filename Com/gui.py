#!/usr/bin/env python3
import threading, queue, time, io
from datetime import datetime
from pathlib import Path
from tkinter import Tk, Label, Button, Scale, HORIZONTAL, IntVar, DoubleVar, StringVar, Frame
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk
from pantilt_client import PanTiltClient

DEFAULT_HOST = "raspberrypi.local"
DEFAULT_PORT = 7000
PREVIEW_W, PREVIEW_H = 480, 360   # 고정 프리뷰 크기(버튼 밀림 방지)

class StreamFetcher(threading.Thread):
    """ /stream MJPEG 파서 (백그라운드) """
    def __init__(self, cli: PanTiltClient, out_q: queue.Queue):
        super().__init__(daemon=True)
        self.cli, self.out_q, self.running = cli, out_q, True
    def run(self):
        try:
            r = self.cli.sess.get(self.cli.base + "/stream", stream=True, timeout=10)
            r.raise_for_status()
            buf = b""
            for chunk in r.iter_content(1024):
                if not self.running: break
                if not chunk: continue
                buf += chunk
                while True:
                    st = buf.find(b'\xff\xd8'); en = buf.find(b'\xff\xd9')
                    if st != -1 and en != -1 and en > st:
                        jpg = buf[st:en+2]; buf = buf[en+2:]
                        self.out_q.put(("frame", jpg))
                    else:
                        break
        except Exception as e:
            self.out_q.put(("error", f"스트림 에러: {e}"))
    def stop(self): self.running = False

class ScanWorker(threading.Thread):
    def __init__(self, cli: PanTiltClient, cfg: dict, out_q: queue.Queue):
        super().__init__(daemon=True)
        self.cli, self.cfg, self.out_q = cli, cfg, out_q
        self.stop_flag = False
    def stop(self): self.stop_flag = True
    def run(self):
        try:
            base = Path(self.cfg["outdir"]); base.mkdir(parents=True, exist_ok=True)
            pans  = self._irange(self.cfg["pan_min"],  self.cfg["pan_max"],  self.cfg["pan_step"])
            tilts = self._irange(self.cfg["tilt_min"], self.cfg["tilt_max"], self.cfg["tilt_step"])
            jobs, done = len(pans)*len(tilts), 0
            self.out_q.put(("scan_start", jobs))
            for t in tilts:
                for p in pans:
                    if self.stop_flag: raise InterruptedError
                    self.cli.move(p, t, speed=self.cfg["speed"], acc=self.cfg["acc"])
                    time.sleep(self.cfg["settle"])
                    ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
                    name = f"img_t{t:+03d}_p{p:+04d}_{ts}.jpg"
                    self.cli.capture_to_file(str(base/name), size=(self.cfg["width"], self.cfg["height"]))
                    done += 1
                    self.out_q.put(("scan_prog", (done, jobs, str(base/name))))
            self.cli.stop(); self.out_q.put(("scan_done", jobs))
        except InterruptedError:
            self.cli.stop(); self.out_q.put(("scan_abort", None))
        except Exception as e:
            self.out_q.put(("scan_err", str(e)))
    @staticmethod
    def _irange(start:int, end:int, step:int):
        if step <= 0: raise ValueError("step must be > 0")
        vals=[]; v=start
        if start<=end:
            while v<=end: vals.append(v); v+=step
        else:
            while v>=end: vals.append(v); v-=step
        return vals

class App:
    def __init__(self, root: Tk):
        root.title("Pan-Tilt Scanner (LAN GUI)")
        root.geometry("980x680")  # 기본 창 크기
        self.root = root
        self.host = StringVar(value=DEFAULT_HOST)
        self.port = IntVar(value=DEFAULT_PORT)
        self.cli = None
        self.q = queue.Queue()
        self.streamer = None
        self.scan_worker = None
        # ---- 레이아웃: 좌(프리뷰) / 우(컨트롤) 2컬럼 ----
        main = Frame(root); main.pack(fill="both", expand=True, padx=10, pady=10)
        left = Frame(main);  left.grid(row=0, column=0, sticky="n")
        right = Frame(main); right.grid(row=0, column=1, sticky="n", padx=(12,0))
        # 좌: 연결/프리뷰
        top = Frame(left); top.pack(fill="x", pady=(0,8))
        Label(top, text="Host/IP:").pack(side="left")
        self.host_entry = ttk.Entry(top, textvariable=self.host, width=24); self.host_entry.pack(side="left", padx=6)
        Button(top, text="Connect", command=self.connect).pack(side="left", padx=4)
        Button(top, text="Center (0,0)", command=self.center).pack(side="left", padx=4)
        Button(top, text="Stop", command=self.stop_move).pack(side="left", padx=4)
        self.preview = Label(left, width=PREVIEW_W, height=PREVIEW_H, bg="#222"); self.preview.pack()
        # 우: 컨트롤/스캔/프로그레스
        ctrl = Frame(right); ctrl.pack(fill="x")
        self.pan=IntVar(value=0); self.tilt=IntVar(value=0)
        self.speed=IntVar(value=60); self.acc=DoubleVar(value=0.5); self.led=IntVar(value=0)  # 기본값 조정(체감 빠르게)
        self._row(ctrl, "Pan",  -180, 180, self.pan,  0)
        self._row(ctrl, "Tilt",  -30,  90, self.tilt, 1)
        self._row(ctrl, "Speed",   0, 100, self.speed,2)
        self._row(ctrl, "Accel",   0,   1, self.acc,  3, 0.1)
        self._row(ctrl, "LED",     0, 255, self.led,  4)
        btns = Frame(ctrl); btns.grid(row=0, column=2, rowspan=3, padx=8)
        Button(btns, text="Apply Move", command=self.apply_move).pack(fill="x", pady=2)
        Button(btns, text="Move FAST",  command=self.apply_move_fast).pack(fill="x", pady=2)
        Button(btns, text="Set LED",    command=self.apply_led).pack(fill="x", pady=2)
        scan = Frame(right); scan.pack(fill="x", pady=(10,6))
        Label(scan, text="Pan min/max/step").grid(row=0, column=0, sticky="w")
        Label(scan, text="Tilt min/max/step").grid(row=1, column=0, sticky="w")
        Label(scan, text="Resolution (w x h)").grid(row=2, column=0, sticky="w")
        Label(scan, text="Settle(s)").grid(row=3, column=0, sticky="w")
        self.pan_min=IntVar(value=-180); self.pan_max=IntVar(value=180); self.pan_step=IntVar(value=15)
        self.tilt_min=IntVar(value=-30); self.tilt_max=IntVar(value=90); self.tilt_step=IntVar(value=15)
        self.width=IntVar(value=1920);   self.height=IntVar(value=1080); self.settle=DoubleVar(value=0.25)
        self._triple(scan, self.pan_min, self.pan_max, self.pan_step, 0)
        self._triple(scan, self.tilt_min, self.tilt_max, self.tilt_step, 1)
        self._triple(scan, self.width,    self.height,   None,          2, ("W","H",""))
        self._single(scan, self.settle, 3, 1, 8)
        outrow = Frame(right); outrow.pack(fill="x", pady=(4,6))
        Button(outrow, text="Select Output Folder", command=self.choose_outdir).pack(side="left")
        self.outdir=StringVar(value=str(Path(f"captures_{datetime.now().strftime('%Y%m%d_%H%M%S')}")))
        Label(outrow, textvariable=self.outdir).pack(side="left", padx=6)
        ops = Frame(right); ops.pack(fill="x", pady=(6,0))
        Button(ops, text="Start Scan", command=self.start_scan).pack(side="left", padx=4)
        Button(ops, text="Stop Scan",  command=self.stop_scan).pack(side="left", padx=4)
        self.prog = ttk.Progressbar(ops, orient=HORIZONTAL, length=260, mode="determinate"); self.prog.pack(side="left", padx=10)
        self.prog_lbl = Label(ops, text="0 / 0"); self.prog_lbl.pack(side="left")
        # 스트림/스캔 이벤트 처리
        self.tkimg=None
        self.root.after(60, self._poll)

    # ----- helpers -----
    def _row(self, parent, name, mn, mx, var, row, res=1):
        Label(parent, text=name).grid(row=row, column=0, sticky="w")
        Scale(parent, from_=mn, to=mx, orient=HORIZONTAL, variable=var, resolution=res, length=260)\
            .grid(row=row, column=1, sticky="ew", padx=6, pady=2)
    def _triple(self, parent, v1, v2, v3, row, labels=("min","max","step")):
        ttk.Entry(parent, width=7, textvariable=v1).grid(row=row, column=1, sticky="w", padx=4)
        ttk.Entry(parent, width=7, textvariable=v2).grid(row=row, column=2, sticky="w", padx=4)
        if v3 is not None: ttk.Entry(parent, width=7, textvariable=v3).grid(row=row, column=3, sticky="w", padx=4)
    def _single(self, parent, v, row, col_start=1, width=10):
        ttk.Entry(parent, width=width, textvariable=v).grid(row=row, column=col_start, sticky="w", padx=4)

    # ----- actions (모두 비동기 실행으로 UI 멈춤 방지) -----
    def _async(self, func, *args, **kw):
        threading.Thread(target=lambda: func(*args, **kw), daemon=True).start()

    def connect(self):
        try:
            self.cli = PanTiltClient(self.host.get(), self.port.get())
            self.cli.health()
        except Exception as e:
            messagebox.showerror("Connect", f"연결 실패: {e}"); return
        self.streamer = StreamFetcher(self.cli, self.q); self.streamer.start()
        messagebox.showinfo("Connect", "연결 성공! 스트림 시작.")

    def center(self):
        if not self.cli: return
        self.pan.set(0); self.tilt.set(0)
        self._async(self.cli.move, 0, 0, self.speed.get(), self.acc.get())

    def stop_move(self):
        if not self.cli: return
        self._async(self.cli.stop)

    def apply_move(self):
        if not self.cli: return
        self._async(self.cli.move, self.pan.get(), self.tilt.get(), self.speed.get(), self.acc.get())

    def apply_move_fast(self):
        if not self.cli: return
        # 체감 빠르게(일부 펌웨어에서 speed=0이 느릴 수 있으니 확실히 큰 값)
        self._async(self.cli.move, self.pan.get(), self.tilt.get(), 100, 1.0)

    def apply_led(self):
        if not self.cli: return
        self._async(self.cli.led, self.led.get())

    def choose_outdir(self):
        d = filedialog.askdirectory()
        if d: self.outdir.set(d)

    def start_scan(self):
        if not self.cli:
            messagebox.showwarning("Scan", "먼저 Connect 해주세요."); return
        if self.scan_worker and self.scan_worker.is_alive():
            messagebox.showwarning("Scan", "이미 스캔 중입니다."); return
        cfg = dict(
            pan_min=self.pan_min.get(), pan_max=self.pan_max.get(), pan_step=self.pan_step.get(),
            tilt_min=self.tilt_min.get(), tilt_max=self.tilt_max.get(), tilt_step=self.tilt_step.get(),
            speed=self.speed.get(), acc=self.acc.get(), settle=float(self.settle.get()),
            width=self.width.get(), height=self.height.get(), outdir=self.outdir.get()
        )
        self.scan_worker = ScanWorker(self.cli, cfg, self.q); self.scan_worker.start()

    def stop_scan(self):
        if self.scan_worker and self.scan_worker.is_alive(): self.scan_worker.stop()

    # ----- queue poll -----
    def _poll(self):
        try:
            while True:
                tag, payload = self.q.get_nowait()
                if tag == "frame":
                    img = Image.open(io.BytesIO(payload))
                    img.thumbnail((PREVIEW_W, PREVIEW_H))
                    self.tkimg = ImageTk.PhotoImage(img)
                    self.preview.configure(image=self.tkimg, width=PREVIEW_W, height=PREVIEW_H)
                elif tag == "error":
                    print(payload)
                elif tag == "scan_start":
                    total = payload; self.prog.configure(maximum=total, value=0); self.prog_lbl.config(text=f"0 / {total}")
                elif tag == "scan_prog":
                    done, total, path = payload
                    self.prog.configure(value=done); self.prog_lbl.config(text=f"{done} / {total}")
                elif tag == "scan_done":
                    total = payload; messagebox.showinfo("Scan", f"스캔 완료! 총 {total}장 저장됨.")
                elif tag == "scan_abort":
                    messagebox.showinfo("Scan", "사용자 중지.")
                elif tag == "scan_err":
                    messagebox.showerror("Scan", f"오류: {payload}")
        except queue.Empty:
            pass
        self.root.after(60, self._poll)

def main():
    root = Tk()
    App(root)
    root.mainloop()

if __name__ == "__main__":
    main()
