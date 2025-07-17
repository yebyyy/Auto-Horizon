"""
Click-to-collect HSV values from selected sample frames.

Typical Workflow
----------------
1. Dump a batch of raw frames first

   $ DUMP_FRAMES_N=300 python -m scripts.capture

   → 300 PNGs appear in samples/frame_000.png … frame_299.png

2. Launch this helper and specify exactly which frames you want to label
   (either by index **or** by full path):

   $ python -m scripts.pick_hsv 1 22 220 231  45 77 99 150 175 299
   # or
   $ python -m scripts.pick_hsv samples/frame_0231.png samples/frame_0045.png

   •  Left-click arrow pixels; press **Q** to move to the next image.
   •  For each image you’re prompted to type which colour bucket the
      pixels belong to:  blue  / yellow / orange / red  (just the
      first 3 letters are enough).

3. At the end you get min / max HSV print-outs for every bucket,
   ready to paste into capture.py.
"""

from __future__ import annotations
import sys, glob, pathlib, collections, keyboard
import cv2
import numpy as np

# ----------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------
SAMPLES_DIR = pathlib.Path("samples")     # where frame_000.png live
ALLOWED_BUCKETS = {
    "blu": "blue",
    "yel": "yellow",
    "ora": "orange",
    "red": "red"
}

# ----------------------------------------------------------------------
# Mouse-click callback
# ----------------------------------------------------------------------
hsv_buckets: dict[str, list[np.ndarray]] = collections.defaultdict(list)

def click_cb(event, x, y, flags, userdata):
    if event == cv2.EVENT_LBUTTONDOWN:
        hsv_val = userdata["hsv"][y, x]              # [H,S,V]
        bucket   = userdata["bucket"]
        hsv_buckets[bucket].append(hsv_val)
        print(f"{bucket:<6}  H={hsv_val[0]:4d}  S={hsv_val[1]:4d}  V={hsv_val[2]:4d}")

# ----------------------------------------------------------------------
# 1. Resolve which images to annotate
# ----------------------------------------------------------------------
cli_args = sys.argv[1:]                # numbers or file names
if not cli_args:
    # No args → default to every PNG in samples/
    target_paths = sorted(SAMPLES_DIR.glob("*.png"))
else:
    resolved = []
    for arg in cli_args:
        if arg.isdigit():              # "22" → samples/frame_022.png
            idx = int(arg)
            resolved.append(SAMPLES_DIR / f"frame_{idx:04d}.png")
        else:                          # assume explicit path
            resolved.append(pathlib.Path(arg))
    target_paths = resolved

# Filter out missing files
target_paths = [p for p in target_paths if p.is_file()]
if not target_paths:
    sys.exit("❌  No matching sample images found.")

print(f"Will annotate {len(target_paths)} image(s):")
for p in target_paths:
    print(" •", p.name)
print()

# ----------------------------------------------------------------------
# 2. Interactive annotation loop
# ----------------------------------------------------------------------
for img_path in target_paths:
    rgb = cv2.imread(str(img_path))
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)

    # Prompt for colour bucket
    ans = input(
        f"{img_path.name}  bucket? "
        "(blu / yel / ora / red, ENTER to skip) ➜ "
    ).strip().lower()[:3]

    if ans not in ALLOWED_BUCKETS:
        print("skipped\n")
        continue

    bucket_key = ALLOWED_BUCKETS[ans]
    print(f"→ Selected bucket: {bucket_key}.  Click pixels, press Q when done.")
    cv2.namedWindow("img")
    cv2.setMouseCallback("img", click_cb,
                         {"hsv": hsv, "bucket": bucket_key})

    while True:
        cv2.imshow("img", rgb)
        if cv2.waitKey(10) & keyboard.is_pressed("q"):
            break

    cv2.destroyAllWindows()
    print()

# ----------------------------------------------------------------------
# 3. Print aggregate ranges
# ----------------------------------------------------------------------
print("="*48, "\nSummary:")
for key, arr in hsv_buckets.items():
    arr = np.stack(arr, axis=0)  # N×3
    h_min, s_min, v_min = arr.min(0)
    h_max, s_max, v_max = arr.max(0)
    print(f"{key:<6}  "
          f"H[{h_min:3d},{h_max:3d}]  "
          f"S[{s_min:3d},{s_max:3d}]  "
          f"V[{v_min:3d},{v_max:3d}]  "
          f"N={len(arr)}")

print("\nDone.  Copy the above H-S-V ranges into capture.py ✂️")
