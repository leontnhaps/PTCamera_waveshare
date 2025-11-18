# processors/undistort_processor.py
import cv2
import numpy as np
from pathlib import Path

try:
    import torch
    import torch.nn.functional as F
    _TORCH_AVAILABLE = True
except Exception:
    torch = None
    F = None
    _TORCH_AVAILABLE = False


class UndistortProcessor:
    """카메라 왜곡 보정 처리기 (CPU/CUDA/Torch 지원)"""
    
    def __init__(self):
        # 보정 파라미터
        self._model = None
        self._K = None
        self._D = None
        self._img_size = None
        self._src_size = None
        
        # OpenCV 맵들
        self._m1 = None
        self._m2 = None
        
        # CUDA 관련
        self._use_cv2_cuda = False
        self._gm1 = None
        self._gm2 = None
        
        # Torch 관련
        self._torch_available = _TORCH_AVAILABLE
        self._torch_cuda = bool(_TORCH_AVAILABLE and torch.cuda.is_available())
        self._torch_device = torch.device("cuda") if self._torch_cuda else torch.device("cpu") if _TORCH_AVAILABLE else None
        self._torch_use_fp16 = False
        self._torch_dtype = (torch.float16 if (self._torch_cuda and self._torch_use_fp16) else torch.float32) if _TORCH_AVAILABLE else None
        self._torch_grid = None
        self._torch_grid_wh = None
        
        # CUDA 지원 체크
        try:
            self._use_cv2_cuda = hasattr(cv2, "cuda") and cv2.cuda.getCudaEnabledDeviceCount() > 0
        except Exception:
            self._use_cv2_cuda = False
            
        print(f"[UndistortProcessor] cv2.cuda={self._use_cv2_cuda}, torch_cuda={self._torch_cuda}")
    
    def load_calibration(self, npz_path: str) -> bool:
        """보정 파일(.npz) 로드"""
        try:
            cal = np.load(npz_path, allow_pickle=True)
            self._model = str(cal["model"])
            self._K = cal["K"].astype(np.float32)
            self._D = cal["D"].astype(np.float32)
            self._img_size = tuple(int(x) for x in cal["img_size"])
            
            # 기존 맵들 초기화
            self._src_size = None
            self._m1 = self._m2 = None
            self._gm1 = self._gm2 = None
            self._torch_grid = None
            self._torch_grid_wh = None
            
            print(f"[UndistortProcessor] Loaded: model={self._model}, img_size={self._img_size}")
            return True
        except Exception as e:
            print(f"[UndistortProcessor] Load failed: {e}")
            return False
    
    def is_loaded(self) -> bool:
        """보정 파라미터가 로드되었는지 확인"""
        return self._K is not None and self._D is not None and self._model is not None
    
    def process(self, bgr: np.ndarray, alpha: float = 0.0) -> np.ndarray:
        """
        이미지 왜곡 보정
        우선순위: Torch CUDA → cv2 CUDA → CPU
        """
        if not self.is_loaded():
            return bgr
            
        h, w = bgr.shape[:2]
        self._ensure_maps(w, h, alpha)
        
        # Torch CUDA 경로
        if self._torch_cuda and self._m1 is not None:
            try:
                self._ensure_torch_grid(w, h)
                if self._torch_grid is not None:
                    return self._undistort_torch(bgr)
            except Exception as e:
                print(f"[UndistortProcessor][torch] Failed, fallback: {e}")
        
        # cv2 CUDA 경로
        if self._use_cv2_cuda and self._gm1 is not None and self._gm2 is not None:
            try:
                return self._undistort_cuda(bgr)
            except Exception as e:
                print(f"[UndistortProcessor][cv2.cuda] Failed, fallback: {e}")
        
        # CPU 경로
        return self._undistort_cpu(bgr)
    
    def _scale_K(self, K, sx, sy):
        """카메라 매트릭스 스케일링"""
        K2 = K.copy()
        K2[0,0] *= sx; K2[1,1] *= sy
        K2[0,2] *= sx; K2[1,2] *= sy
        K2[2,2] = 1.0
        return K2
    
    def _ensure_maps(self, w: int, h: int, alpha: float):
        """언디스토트 맵 생성 (필요시에만)"""
        if self._src_size == (w, h) and self._m1 is not None:
            return
            
        Wc, Hc = self._img_size
        sx, sy = w / float(Wc), h / float(Hc)
        K = self._scale_K(self._K, sx, sy)
        D = self._D
        
        if self._model == "pinhole":
            newK, _ = cv2.getOptimalNewCameraMatrix(K, D, (w, h), alpha=alpha, newImgSize=(w, h))
            m1, m2 = cv2.initUndistortRectifyMap(K, D, None, newK, (w, h), cv2.CV_16SC2)
        else:
            R = np.eye(3, dtype=np.float32)
            newK = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(
                K, D, (w, h), R, balance=alpha, new_size=(w, h)
            )
            m1, m2 = cv2.fisheye.initUndistortRectifyMap(K, D, R, newK, (w, h), cv2.CV_16SC2)
        
        self._m1, self._m2 = m1, m2
        self._src_size = (w, h)
        
        # CUDA 맵 업로드
        if self._use_cv2_cuda:
            try:
                self._gm1 = cv2.cuda_GpuMat()
                self._gm1.upload(self._m1)
                self._gm2 = cv2.cuda_GpuMat()
                self._gm2.upload(self._m2)
            except Exception as e:
                print(f"[UndistortProcessor] CUDA map upload failed: {e}")
                self._gm1 = self._gm2 = None
        
        # Torch grid 무효화
        self._torch_grid = None
        self._torch_grid_wh = None
    
    def _ensure_torch_grid(self, w: int, h: int):
        """Torch용 정규화된 그리드 생성"""
        if not (self._torch_cuda and self._m1 is not None):
            return
        if self._torch_grid is not None and self._torch_grid_wh == (w, h):
            return
            
        mx, my = cv2.convertMaps(self._m1, self._m2, cv2.CV_32F)
        H, W = mx.shape
        gx = (mx / max(W-1, 1)) * 2.0 - 1.0
        gy = (my / max(H-1, 1)) * 2.0 - 1.0
        grid = np.stack([gx, gy], axis=-1)
        
        self._torch_grid = torch.from_numpy(grid).unsqueeze(0).to(
            device=self._torch_device, dtype=self._torch_dtype
        )
        self._torch_grid_wh = (w, h)
    
    def _undistort_torch(self, bgr: np.ndarray) -> np.ndarray:
        """Torch CUDA로 언디스토트"""
        t_cpu = torch.from_numpy(bgr).permute(2, 0, 1).contiguous()
        try:
            t_cpu = t_cpu.pin_memory()
        except Exception:
            pass
        
        t = t_cpu.to(self._torch_device, dtype=self._torch_dtype, non_blocking=True).unsqueeze(0) / 255.0
        out = F.grid_sample(t, self._torch_grid, mode="bilinear", align_corners=True)
        result = (out.squeeze(0).permute(1, 2, 0) * 255.0).clamp(0, 255).byte().cpu().numpy()
        return np.ascontiguousarray(result)
    
    def _undistort_cuda(self, bgr: np.ndarray) -> np.ndarray:
        """OpenCV CUDA로 언디스토트"""
        gsrc = cv2.cuda_GpuMat()
        gsrc.upload(bgr)
        gout = cv2.cuda.remap(gsrc, self._gm1, self._gm2,
                              interpolation=cv2.INTER_LINEAR, 
                              borderMode=cv2.BORDER_CONSTANT)
        return gout.download()
    
    def _undistort_cpu(self, bgr: np.ndarray) -> np.ndarray:
        """CPU로 언디스토트"""
        return cv2.remap(bgr, self._m1, self._m2, cv2.INTER_LINEAR)