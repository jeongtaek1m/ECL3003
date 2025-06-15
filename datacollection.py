import cv2
import time
import os
import shutil
import numpy as np
import json
import sys, tty, termios, select
import argparse
from base_ctrl import BaseController
from jetcam.csi_camera import CSICamera

# ───── 명령줄 인자 ─────
parser = argparse.ArgumentParser()
parser.add_argument("--save_dataset", action="store_true", help="학습용 데이터 저장")
parser.add_argument("--save_debug", action="store_true", help="디버깅 이미지 저장")
args = parser.parse_args()

# ───── 설정값 ─────
CAMERA_WIDTH = 960
CAMERA_HEIGHT = 540
FPS = 10 
RECORD_INTERVAL = 0.1

DATASETS_DIR = "./datasets/"
TMP_DATASET_DIR = os.path.join(DATASETS_DIR, "tmp/")
ANNOTATIONS_FILE = "annotations.csv"

DEBUG_BASE_DIR = "./debug_images/"
DEBUG_IMG_DIR = os.path.join(DEBUG_BASE_DIR, "tmp/")

# ───── 디렉토리 초기화 ─────
def reset_temp_dirs():
    if args.save_dataset:
        if os.path.exists(TMP_DATASET_DIR):
            shutil.rmtree(TMP_DATASET_DIR)
        os.makedirs(TMP_DATASET_DIR)
    if args.save_debug:
        if os.path.exists(DEBUG_IMG_DIR):
            shutil.rmtree(DEBUG_IMG_DIR)
        os.makedirs(DEBUG_IMG_DIR)

reset_temp_dirs()

# ───── 장치 초기화 ─────
camera = CSICamera(capture_width=CAMERA_WIDTH, capture_height=CAMERA_HEIGHT, capture_fps=FPS)
camera.running = True

controller = BaseController("/dev/ttyUSB0", 115200)
controller.current_left = 0.0
controller.current_right = 0.0

def set_diff_speed(left, right):
    controller.current_left = left
    controller.current_right = right
    controller.base_speed_ctrl(left, right)

# ───── 비차단 키보드 설정 ─────
def init_keyboard():
    fd = sys.stdin.fileno()
    global old_term
    old_term = termios.tcgetattr(fd)
    tty.setcbreak(fd)

def restore_keyboard():
    termios.tcsetattr(sys.stdin.fileno(), termios.TCSADRAIN, old_term)

def kbhit():
    return select.select([sys.stdin], [], [], 0)[0]

def getch_nonblock():
    if kbhit():
        return sys.stdin.read(1)
    return None

# ───── 주행 + 저장 함수 ─────
def save_annotated_camera_frame(frame):
    timestamp = str(int(time.time() * 1000))

    if args.save_dataset:
        raw_path = os.path.join(TMP_DATASET_DIR, timestamp + ".jpg")
        cv2.imwrite(raw_path, frame)
        with open(os.path.join(TMP_DATASET_DIR, ANNOTATIONS_FILE), 'a') as f:
            f.write(f"{timestamp}, {controller.current_left:.2f}, {controller.current_right:.2f}\n")

    if args.save_debug:
        annotated = frame.copy()
        text = f"L: {controller.current_left:.2f}, R: {controller.current_right:.2f}"
        cv2.putText(annotated, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                    1.0, (0, 255, 0), 2, cv2.LINE_AA)
        debug_path = os.path.join(DEBUG_IMG_DIR, timestamp + ".jpg")
        cv2.imwrite(debug_path, annotated)

# ───── 메인 루프 ─────
print("WASD: 이동 / E: 천천히 전진 / SPACE: 정지 / Q: 종료")

last_record_time = time.time()

try:
    init_keyboard()

    while True:
        key = getch_nonblock()

        if key is not None:
            L, R = controller.current_left, controller.current_right

            if key == ' ':
                set_diff_speed(0.0, 0.0)
                break
            elif key == 'w':
                L = 0.25
                R = 0.25*1.03
            elif key == 's':
                L = 0.1
                R = 0.1
            elif key == 'a':
                L = 0.03
                R = 0.43*1.03
                # R = 0.5
            elif key == 'd':
                L = 0.43
                # L = 0.5
                R = 0.05*1.03
                # R = 0.03
            elif key == 'q':
                L = 0.05
                R = 0.4*1.03
            elif key == 'e':
                L = 0.4
                R = 0.05*1.03

            L = np.clip(L, -0.5, 0.5)
            R = np.clip(R, -0.5, 0.5)
            set_diff_speed(L, R)

        now = time.time()
        if now - last_record_time >= RECORD_INTERVAL:
            frame = camera.value
            save_annotated_camera_frame(frame)
            last_record_time = now
            # print(f"Saved: L={controller.current_left:.2f}, R={controller.current_right:.2f}")

        time.sleep(0.01)

except KeyboardInterrupt:
    print("Interrupted")
    set_diff_speed(0.0, 0.0)

finally:
    restore_keyboard()
    def save_recording():
        session_name = "diffdrive_" + time.strftime("%Y%m%d_%H%M%S")
        if args.save_dataset:
            save_dir = os.path.join(DATASETS_DIR, session_name)
            os.makedirs(save_dir, exist_ok=True)
            for file in os.listdir(TMP_DATASET_DIR):
                shutil.move(os.path.join(TMP_DATASET_DIR, file), os.path.join(save_dir, file))
            print(f"✅ 학습 데이터 저장됨: {save_dir}")

        if args.save_debug:
            debug_dir = os.path.join(DEBUG_BASE_DIR, session_name)
            os.makedirs(debug_dir, exist_ok=True)
            for file in os.listdir(DEBUG_IMG_DIR):
                shutil.move(os.path.join(DEBUG_IMG_DIR, file), os.path.join(debug_dir, file))
            print(f"🟩 디버깅 이미지 저장됨: {debug_dir}")

    save_recording()
