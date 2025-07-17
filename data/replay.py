import numpy as np, cv2, time, os, sys

data  = np.load("data/demos/250707-122710-road-fall-911-gt3.npz")
obs   = data["obs"]      # (N, S, C, H, W)
acts  = data["act"]      # (N, 3)

S, C, H, W = obs.shape[1:]

def split_rgb_mask(frame_C_H_W):
    """
    Input  C×H×W  (C==4) – returns (rgb, mask)
    rgb  : H×W×3  uint8
    mask : H×W    uint8  (0 or 255)
    """
    rgb  = (frame_C_H_W[:3] * 255).astype(np.uint8)          # 3×H×W
    rgb  = np.transpose(rgb, (1, 2, 0))                      # H×W×3
    m    = (frame_C_H_W[3] * 255).astype(np.uint8)           # H×W
    return rgb, m

for i in range(len(obs)):
    frame = obs[i, -1]                   # latest frame in stack
    if C == 3:
        img   = (frame * 255).astype("uint8")
        img = img.transpose(1, 2, 0)     
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    elif C == 4:                    # new demo with mask channel
        rgb, mask = split_rgb_mask(frame)
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

        # build visualisation:   left = colour  |  right = coloured mask
        mask_vis = cv2.applyColorMap(mask, cv2.COLORMAP_JET)
        vis = np.hstack([bgr, mask_vis])

    elif C == 1:
        img = img.squeeze(0)            

    else:
         sys.exit(f"Unexpected C={C}; expected 3 or 4")
        

    cv2.imshow("Replay", img)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
    time.sleep(1/30)                    
    
cv2.destroyAllWindows()