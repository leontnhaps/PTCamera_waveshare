# config.py
from pathlib import Path
from datetime import datetime

# ============== 네트워크 설정 ==============
# GUI에서 서버로 연결할 때 사용
GUI_SERVER_HOST = "127.0.0.1"  # 원래 pc_gui.py의 SERVER_HOST
GUI_CTRL_PORT = 7600            # 원래 pc_gui.py의 GUI_CTRL_PORT  
GUI_IMG_PORT = 7601             # 원래 pc_gui.py의 GUI_IMG_PORT

# Pi에서 서버로 연결할 때 사용  
PI_SERVER_HOST_711A = "192.168.0.9"   # 711a (주석 그대로 유지)
PI_SERVER_HOST_602A = "172.30.1.100"  # 602a (현재 사용중)
PI_SERVER_HOST_hotspot = "10.95.38.118" # hotspot (테스트용)
PI_SERVER_HOST = PI_SERVER_HOST_hotspot # 현재 사용할 IP

# 서버에서 사용하는 포트들
AGENT_CTRL_PORT = 7500         # 원래 pc_server.py의 AGENT_CTRL_PORT
AGENT_IMG_PORT = 7501          # 원래 pc_server.py의 AGENT_IMG_PORT


# ============== 카메라/하드웨어 설정 ==============
SERIAL_BAUD = 115200           # 원래 pi_agent.py의 BAUD
MAX_WIDTH = 2592               # 원래 pi_agent.py의 MAX_W
MAX_HEIGHT = 1944              # 원래 pi_agent.py의 MAX_H


# ============== 폴더 설정 ==============
def get_gui_output_dir():
    """GUI 캡처 폴더 (원래 DEFAULT_OUT_DIR)"""
    return Path(f"captures_gui_{datetime.now().strftime('%Y%m%d_%H%M%S')}")

def get_server_output_dir():  
    """서버 캡처 폴더"""
    return Path(f"captures_server_{datetime.now().strftime('%Y%m%d_%H%M%S')}")

class AutoLoadPaths:
    """시작시 자동으로 로드할 파일들의 경로"""
    
    # ✅ 실제 파일 경로로 바꿔주세요!
    CALIB_NPZ_PATH = r"C:\Users\gmlwn\OneDrive\바탕 화면\ICon1학년\OpticalWPT\PTCamera_waveshare\calib.npz"
    YOLO_WEIGHTS_PATH = r"C:\Users\gmlwn\OneDrive\바탕 화면\ICon1학년\OpticalWPT\PTCamera_waveshare\yolov11m.pt"
    
    # 자동 로드 활성화/비활성화 (False로 하면 기존처럼 수동 선택)
    AUTO_LOAD_CALIB = True   # True = 자동 로드, False = 수동 선택
    AUTO_LOAD_YOLO = True    # True = 자동 로드, False = 수동 선택

# 전역 인스턴스 생성
auto_load = AutoLoadPaths()