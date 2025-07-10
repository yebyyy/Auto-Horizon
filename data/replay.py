import numpy as np, cv2, time, os

data  = np.load("data/demos/250707-122710-road-fall-911-gt3.npz")
obs   = data["obs"]      # (N, S, C, H, W)
acts  = data["act"]      # (N, 3)

S, C, H, W = obs.shape[1:]

for i in range(len(obs)):
    frame = obs[i, -1]                   # latest frame in stack
    img   = (frame * 255).astype("uint8")

    if C == 1:
        img = img.squeeze(0)             
    else:
        img = img.transpose(1, 2, 0)     
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    cv2.imshow("Replay", img)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
    time.sleep(1/30)                    
    
cv2.destroyAllWindows()