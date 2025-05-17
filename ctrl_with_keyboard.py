from base_ctrl import BaseController
import time
from pynput import keyboard
import threading

# === Constants ===
FORWARD_SPEED = -100
BACKWARD_SPEED = 100
TURN_RATIO = 0.5  # 조향 시 감속 비율
UPDATE_INTERVAL = 0.1

# === Initialization ===
base = BaseController('/dev/ttyUSB0', 115200)

# === Direction Flags ===
forward = False
backward = False
left = False
right = False
ccw = False
cw = False

def send_control_async(L, R):
    def worker():
        base.base_json_ctrl({"T": 11, "L": L, "R": R})
    threading.Thread(target=worker).start()

def on_press(key):
    global forward, backward, left, right, ccw, cw
    try:
        if key.char == 'w':
            forward = True
        elif key.char == 's':
            backward = True
        elif key.char == 'a':
            left = True
        elif key.char == 'd':
            right = True
        elif key.char == 'q':
            ccw = True
        elif key.char == 'e':
            cw = True
    except:
        pass

def on_release(key):
    global forward, backward, left, right, ccw, cw
    try:
        if key.char == 'w':
            forward = False
        elif key.char == 's':
            backward = False
        elif key.char == 'a':
            left = False
        elif key.char == 'd':
            right = False
        elif key.char == 'q':
            ccw = False
        elif key.char == 'e':
            cw = False
    except:
        pass

def update_vehicle_motion():
    # 기본 속도 설정
    if forward and not backward:
        speed = FORWARD_SPEED
    elif backward and not forward:
        speed = BACKWARD_SPEED
    else:
        speed = 0

    # 조향 설정
    L = speed
    R = speed

    if speed > 0:  # 후진 중 조향
        if left and not right:
            L = 30
            R = 255
        elif right and not left:
            R = 30
            L = 255
        
    elif speed < 0:  # 전진 중 조향 
        if left and not right:
            L = -30
            R = -255
        elif right and not left:
            R = -30
            L = -255

    else:
        if ccw and not cw:
            L = 200
            R = -200
        elif cw and not ccw:
            L = -200
            R = 200

    send_control_async(-L, -R)
    print(f"[UGV] FWD:{forward} BWD:{backward} LEFT:{left} RIGHT:{right} → L: {-L}, R: {-R}")
    print(f"CCW{ccw} CW{cw}-> L: {-L}, R: {-R}")

# === Keyboard Listener ===
listener = keyboard.Listener(on_press=on_press, on_release=on_release)
listener.start()

# === Main Loop ===
try:
    last_update_time = 0
    while listener.running:
        now = time.time()
        if now - last_update_time >= UPDATE_INTERVAL:
            update_vehicle_motion()
            last_update_time = now
except KeyboardInterrupt:
    print("\nQuit")
    base.base_json_ctrl({"T": 11, "L": 0.0, "R": 0.0})
