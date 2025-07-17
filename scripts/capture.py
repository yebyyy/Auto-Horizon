# scripts/capture.py
import dxcam
import numpy as np
import cv2
import torch
from collections import deque
import time
import win32gui

# ─── CONFIGURE THIS ──────────────────────────────────────
FRAME_SIZE = (224, 144)   # (W, H) for the network
STACK_SIZE = 4
CYAN_LO  = ( 80,  40,  60);  CYAN_HI  = (105, 255, 255)   # blue / green
RED1_LO = (105,  80,  60);  RED1_HI = (130, 255, 255)    # orange‑red
RED2_LO = (  0,  80,  60);  RED2_HI = ( 15, 255, 255)   # red

# 1) Create a camera (no target_fps/region in v0.0.5)
camera = dxcam.create(output_color="RGB")

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

GRAY = False  # True for grayscale, False for RGB

# ─── Frame stack buffer ──────────────────────────────────
_stack = deque(
    [torch.zeros(1 if GRAY else 4, FRAME_SIZE[1], FRAME_SIZE[0])] * STACK_SIZE,
    maxlen=STACK_SIZE
)

# ─── Helper functions ───────────────────────────────────
def get_frame(gray=GRAY) -> torch.Tensor:
    """Grab, preprocess, return tensor in [0,1], shape [C,H,W]."""
    img = camera.get_latest_frame()        # numpy H×W×3 RGB
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)

    if img is None:                        # very first call can be None
        time.sleep(0.01)
        return get_frame(gray)

    if gray:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)   # H×W
    # else:
    #     img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)    # H×W×3

    mask = cv2.inRange(hsv, CYAN_LO, CYAN_HI) | cv2.inRange(hsv, RED1_LO, RED1_HI) | cv2.inRange(hsv, RED2_LO, RED2_HI)
    mask = mask.astype(np.float32) / 255.0  # H×W mask in [0,1]
    frame = np.dstack([img, mask])  # H×W×4
    img = cv2.resize(frame, FRAME_SIZE, interpolation=cv2.INTER_AREA)
    tensor = torch.from_numpy(img).float().div(255.0)

    if gray:
        return tensor.unsqueeze(0)        # [1,H,W]
    else:
        return tensor.permute(2, 0, 1)    # [4,H,W]

def get_frame_stack() -> torch.Tensor:
    f = get_frame(gray=GRAY)
    _stack.append(f)
    return torch.stack(list(_stack), dim=0)  # [STACK_SIZE,C,H,W]

# ─── Smoke-test ──────────────────────────────────────────
if __name__ == "__main__":
    DUMP = 1
    if DUMP == 1:
        import imageio, itertools, os
        os.makedirs("samples", exist_ok=True)
        for i in itertools.count():
            rgb_t = get_frame(gray=False)
            image = (rgb_t.permute(1, 2, 0).numpy() * 255).astype(np.uint8)  # H×W×C
            imageio.imwrite(f"samples/frame_{i:04d}.png", image)
            if i == 1000: break  # save 100 frames
        print("dumped frames to 'samples/'")
        quit()
    print("Press Q in the preview window to quit.")
    while True:
        latest = (get_frame_stack()[-1].numpy() * 255).astype(np.uint8)
        if not GRAY:
            latest = np.transpose(latest, (1, 2, 0))  # H×W×C for cv2.imshow
        cv2.imshow("FH4 Capture", latest)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    cv2.destroyAllWindows()
    camera.stop()
