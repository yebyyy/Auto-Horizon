import os, time, datetime, numpy as np
from capture import get_frame_stack
import XInput as xinput# For Xbox controller input

SAVE_DIR = 'data/demos'
os.makedirs(SAVE_DIR, exist_ok=True)


def poll_action(state):
    steer = xinput.get_thumb_values(state)[0][0]  # we only need the X-axis for steering
    throttle = xinput.get_trigger_values(state)[1]
    brake = xinput.get_trigger_values(state)[0]
    return np.array([steer, throttle, brake], dtype=np.float32)

run_id = datetime.datetime.now().strftime("%y%m%d-%H%M%S")
buf_obs, buf_act = [], []

started = False
while True:
    state = xinput.get_state(0)
    if xinput.get_button_values(state)['A'] and not started:
        started = True
        start = time.perf_counter()
        print("Recording started...")
    if xinput.get_button_values(state)['B'] and started:
        print("Recording stopped.")
        break
    buf_obs.append(get_frame_stack().numpy())
    buf_act.append(poll_action(state))
    time.sleep(1/30)  # Poll at 30Hz


elapsed = time.perf_counter() - start
out = os.path.join(SAVE_DIR, f"{run_id}.npz")
np.savez_compressed(out, obs=np.stack(buf_obs), act=np.stack(buf_act))  # Save as compressed npz
print(f"Saved {len(buf_obs)} steps, elapsed {elapsed:.1f}s to {out}")