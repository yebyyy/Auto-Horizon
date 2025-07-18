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
CYAN_LO  = (114, 170, 60);  CYAN_HI  = (120, 220, 170)   # light blue / cyan
YELLOW_LO = (90, 120, 70); YELLOW_HI = (105, 170, 170)   # yellow
ORANGE_LO = (105, 130, 80); ORANGE_HI = (115, 200, 180)  # orange
RED1_LO = (0, 65, 70);  RED1_HI = (15, 165, 140)    # red
RED2_LO = (170, 30, 70);  RED2_HI = (180, 60, 100)   # red

# 1) Create a camera (no target_fps/region in v0.0.5)
camera = dxcam.create(output_color="BGR")

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
    img = camera.get_latest_frame()        # numpy H×W×3 BGR
    img = cv2.resize(img, FRAME_SIZE, interpolation=cv2.INTER_AREA)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    if img is None:                        # very first call can be None
        time.sleep(0.01)
        return get_frame(gray)

    if gray:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)   # H×W
    # else:
    #     img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)    # H×W×3

    mask = (cv2.inRange(hsv, CYAN_LO, CYAN_HI) | cv2.inRange(hsv, RED1_LO, RED1_HI) | cv2.inRange(hsv, RED2_LO, RED2_HI))
    mask = mask.astype(np.float32) / 255.0  # H×W, float in [0,1]
    img = img.astype(np.float32) / 255.0  # H×W or H×W×3, float in [0,1]
    frame = np.dstack([img, mask[..., np.newaxis]])
    tensor = torch.from_numpy(frame).float()

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
    DUMP = 1  # capturing hsv, otherwise 0
    if DUMP == 1:
        import imageio, itertools, os
        os.makedirs("samples", exist_ok=True)
        for i in itertools.count():
            bgr_t = get_frame(gray=False)[:3]
            image = (bgr_t.permute(1, 2, 0).numpy() * 255).astype(np.uint8)  # H×W×C
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
