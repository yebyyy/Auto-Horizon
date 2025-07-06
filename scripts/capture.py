# scripts/capture.py
import dxcam
import numpy as np
import cv2
import torch
from collections import deque
import time
import win32gui

# ─── CONFIGURE THIS ──────────────────────────────────────
FRAME_SIZE = (84, 84)   # (W, H) for the network
STACK_SIZE = 4

# 1) Create a camera (no target_fps/region in v0.0.5)
camera = dxcam.create()

# 2) Locate the FH4 window & get its bounding box
TARGET_TITLE = "Forza Horizon 4"          # exact window title
hwnd = win32gui.FindWindow(None, TARGET_TITLE)
if hwnd == 0:
    raise RuntimeError(f"Window titled '{TARGET_TITLE}' not found. "
                       "Launch FH4 in borderless-window mode first.")

def _window_rect(hwnd):
    # returns (left, top, right, bottom) in desktop coords
    rect = win32gui.GetWindowRect(hwnd)
    return rect

REGION = _window_rect(hwnd)

# 3) Start background capture thread at 30 Hz
camera.start(target_fps=30, region=REGION)

# ─── Frame stack buffer ──────────────────────────────────
_stack = deque(
    [torch.zeros(1, FRAME_SIZE[1], FRAME_SIZE[0])] * STACK_SIZE,
    maxlen=STACK_SIZE
)

# ─── Helper functions ───────────────────────────────────
def get_frame(gray: bool = True) -> torch.Tensor:
    """Grab, preprocess, return tensor in [0,1], shape [C,H,W]."""
    img = camera.get_latest_frame()        # numpy H×W×3 RGB
    if img is None:                        # very first call can be None
        time.sleep(0.01)
        return get_frame(gray)

    if gray:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)   # H×W
    else:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)    # H×W×3

    img = cv2.resize(img, FRAME_SIZE, interpolation=cv2.INTER_AREA)
    tensor = torch.from_numpy(img).float().div(255.0)

    if gray:
        return tensor.unsqueeze(0)        # [1,H,W]
    else:
        return tensor.permute(2, 0, 1)    # [3,H,W]

def get_frame_stack() -> torch.Tensor:
    f = get_frame(gray=True)
    _stack.append(f)
    return torch.cat(list(_stack), dim=0)  # [4,H,W]

# ─── Smoke-test ──────────────────────────────────────────
if __name__ == "__main__":
    print("Press Q in the preview window to quit.")
    while True:
        latest = (get_frame_stack()[-1].numpy() * 255).astype(np.uint8)
        cv2.imshow("FH4 Capture", latest)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    cv2.destroyAllWindows()
    camera.stop()
