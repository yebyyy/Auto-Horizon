import numpy as np, cv2, time, os, sys

data  = np.load("data/demos/250717-233434.npz")
obs   = data["obs"]      # (N, S, C, H, W)
acts  = data["act"]      # (N, 3)

S, C, H, W = obs.shape[1:]

def split_rgb_mask(frame_C_H_W):
    """
    Input  C×H×W  (C==4) – returns (rgb, mask)
    rgb  : H×W×3  uint8
    mask : H×W    uint8  (0 or 255)
    """
    bgr  = (frame_C_H_W[:3] * 255).astype(np.uint8)          # 3×H×W
    bgr  = np.transpose(bgr, (1, 2, 0))                      # H×W×3
    m    = (frame_C_H_W[3] * 255).astype(np.uint8)           # H×W
    return bgr, m

for i in range(len(obs)):
    frame = obs[i, -1]                   # latest frame in stack
    if C == 3:
        img   = (frame * 255).astype("uint8")
        img = img.transpose(1, 2, 0)
        cv2.imshow("Replay BGR", img)

    elif C == 4:                    # new demo with mask channel
        img, mask = split_rgb_mask(frame)

        # build visualisation:   left = colour  |  right = coloured mask
        mask_vis = cv2.applyColorMap(mask, cv2.COLORMAP_JET)
        vis = np.hstack([img, mask_vis])
        cv2.imshow("Replay [BGR | Mask]", vis)

    elif C == 1:
        img = img.squeeze(0)            

    else:
         sys.exit(f"Unexpected C={C}; expected 3 or 4")
        

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
    time.sleep(1/30)                    
    
cv2.destroyAllWindows()