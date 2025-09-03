import requests, base64, pathlib

class PanTiltClient:
    def __init__(self, host: str = "raspberrypi.local", port: int = 7000, timeout: float = 10.0):
        self.base = f"http://{host}:{port}"
        self.timeout = timeout
        self.sess = requests.Session()

    def health(self):
        r = self.sess.get(self.base + "/health", timeout=self.timeout); r.raise_for_status(); return r.json()

    def move(self, pan:int, tilt:int, speed:int=0, acc:float=0.0):
        r = self.sess.post(self.base + "/move",
                           json={"pan":pan, "tilt":tilt, "speed":speed, "acc":acc},
                           timeout=self.timeout); r.raise_for_status(); return r.json()

    def stop(self):
        r = self.sess.post(self.base + "/stop", json={}, timeout=self.timeout); r.raise_for_status(); return r.json()

    def led(self, value:int):
        r = self.sess.post(self.base + "/led", json={"value":value}, timeout=self.timeout); r.raise_for_status(); return r.json()

    def frame_bytes(self) -> bytes:
        r = self.sess.get(self.base + "/frame", timeout=self.timeout)
        r.raise_for_status()
        return r.content

    def capture_to_file(self, out_path: str, size: tuple[int,int] | None = None):
        payload = {}
        if size: payload["size"] = [size[0], size[1]]
        r = self.sess.post(self.base + "/capture", json=payload, timeout=max(self.timeout, 20))
        r.raise_for_status()
        data = r.json()
        img = base64.b64decode(data["image_base64"])
        p = pathlib.Path(out_path); p.write_bytes(img)
        return str(p.resolve())
