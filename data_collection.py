### 전체 통합 코드: DiffDrive + CSICamera + Keyboard + 시간 기반 자동 데이터셋 저장

import cv2
import time
import os
import shutil
import numpy as np
import json
from base_ctrl import BaseController  # UART 기반 로버 제어 클래스

# ───── 설정값 ─────
CAMERA_WIDTH = 960
CAMERA_HEIGHT = 540
FPS = 10
DATASETS_DIR = "./datasets/"
TMP_DATASET_DIR = os.path.join(DATASETS_DIR, "tmp/")
ANNOTATIONS_FILE = "annotations.csv"
RECORD_INTERVAL = 0.1  # 0.1초마다 저장

# ───── 디렉토리 초기화 ─────
def reset_temp_dataset_dir():
    if not os.path.exists(TMP_DATASET_DIR):
        os.makedirs(TMP_DATASET_DIR)
    else:
        shutil.rmtree(TMP_DATASET_DIR)
        os.makedirs(TMP_DATASET_DIR)

reset_temp_dataset_dir()

# ───── 카메라 초기화 ─────
from jetcam.csi_camera import CSICamera
camera = CSICamera(width=CAMERA_WIDTH, height=CAMERA_HEIGHT, capture_fps=FPS)
camera.running = True

# ───── 로버 제어 클래스 초기화 ─────
controller = BaseController("/dev/ttyUSB0", 115200)  # 장치명은 상황에 맞게 수정
controller.current_left = 0.0
controller.current_right = 0.0

# ───── 명령 전달과 값 저장 연결 ─────
def set_diff_speed(left, right):
    controller.current_left = left
    controller.current_right = right
    controller.base_speed_ctrl(left, right)

# ───── 키보드 제어 함수 ─────
import sys, tty, termios

def getch():
    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        ch = sys.stdin.read(1)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)
    return ch

# ───── 주행 중 프레임 + 명령 저장 ─────
def save_annotated_camera_frame(frame):
    timestamp = str(int(time.time()*1000))
    filename = TMP_DATASET_DIR + timestamp + ".jpg"
    
    cv2.imwrite(filename, frame)
    with open(os.path.join(TMP_DATASET_DIR, ANNOTATIONS_FILE), 'a') as f:
        f.write(f"{timestamp}, {round(controller.current_left, 2)}, {round(controller.current_right, 2)}\n")

# ───── 메인 루프 ─────
print("WASD로 제어 / X 정지 / Q 종료 및 저장")

last_record_time = time.time()

try:
    while True:
        key = getch()

        L, R = controller.current_left, controller.current_right
        step = 0.1

        if key == 'q':
            set_diff_speed(0.0, 0.0)
            break
        elif key == 'w':
            L += step
            R += step
        elif key == 's':
            L -= step
            R -= step
        elif key == 'a':
            L -= step
            R += step
        elif key == 'd':
            L += step
            R -= step
        elif key == 'x':
            L, R = 0.0, 0.0

        L = np.clip(L, -1.0, 1.0)
        R = np.clip(R, -1.0, 1.0)
        set_diff_speed(L, R)

        # 시간 기반 자동 저장
        now = time.time()
        if now - last_record_time >= RECORD_INTERVAL:
            frame = camera.value
            save_annotated_camera_frame(frame)
            last_record_time = now
            print(f"Saved: L={L:.2f}, R={R:.2f}")

except KeyboardInterrupt:
    print("Interrupted")
    set_diff_speed(0.0, 0.0)

# ───── 최종 데이터 저장 ─────
def save_recording():
    final_dir = os.path.join(DATASETS_DIR, "diffdrive_" + time.strftime("%Y%m%d_%H%M%S"))
    os.makedirs(final_dir, exist_ok=True)
    for file in os.listdir(TMP_DATASET_DIR):
        shutil.move(os.path.join(TMP_DATASET_DIR, file), os.path.join(final_dir, file))
    print(f"Saved dataset to {final_dir}")

save_recording()
