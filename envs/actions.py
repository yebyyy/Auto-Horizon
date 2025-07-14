import vgamepad as vg
import numpy as np

_PAD = vg.VX360Gamepad()  # create a virtual gamepad instance

def do_action(vec):
    """
    vec: iterable/np.ndarray with 3 floats [steer, gas, brake]
    """

    steer, gas, brake = map(float, vec)
    steer = float(np.clip(steer, -1.0, 1.0))
    gas = float(np.clip(gas, 0.0, 1.0))
    brake = float(np.clip(brake, 0.0, 1.0))

    _PAD.left_joystick_float(x_value_float=steer, y_value_float=0.0) # y_value means up/down, not forward/backward
    _PAD.right_trigger_float(value_float=gas)
    _PAD.left_trigger_float(value_float=brake)
    _PAD.update()  # send the command to the gamepad


if __name__ == "__main__":
    import time
    print("Virtual pad test: steer left→right. Ctrl-C to quit.")
    try:
        while True:
            for s in np.linspace(-1, +1, 41):      # left to right
                do_action([s, 0.5, 0.0])
                time.sleep(0.03)
            for s in np.linspace(+1, -1, 41):      # right to left
                do_action([s, 0.5, 0.0])
                time.sleep(0.03)
            do_action([0, 0, 0.8])  # brake
            time.sleep(3.0)
            do_action([0, 0, 0.0])  # release brake
    except KeyboardInterrupt:
        pass
    finally:
        do_action([0.0, 0.0, 0.0])          # release everything
        print("Released virtual pad.")