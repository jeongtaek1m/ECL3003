"""
Jetson-based 4WD 로버 — 모노 카메라 웨이포인트 추종 스크립트
──────────────────────────────────────────────────────────────
◎ 입력 : CSI 카메라 프레임
◎ 출력 : BaseController JSON 명령(L, R 좌우 바퀴 속도)
◎ 제어 : PID + 좌우 속도 차 감속 (MAX_STEER_RATIO)
"""

# ── 기본 라이브러리 ───────────────────────────────────────────
import time, math, threading
from collections import deque

import cv2
from PIL import Image
import numpy as np
import torch
import torchvision
import matplotlib.pyplot as plt          # (시각화 주석 처리 가능)
import matplotlib.animation as animation  # (시각화 주석 처리 가능)

# ── Jetson / 프로젝트 라이브러리 ──────────────────────────────
from jetcam.csi_camera import CSICamera
from Lane_model.cnn.center_dataset import TEST_TRANSFORMS
from base_ctrl import BaseController

# ── 하이퍼파라미터 / 상수 ────────────────────────────────────
Kp, Ki, Kd       = 0.9, 0.05, 0.2               # PID gains
UPDATE_INTERVAL  = 0.10                        # [s]
BASE_SPEED       = 100                         # 기본 속도 [%]
MAX_STEER_RATIO  = 0.9                         # |steer|=1 일 때 감속비

# ── PID 상태 변수 ────────────────────────────────────────────
_prev_error = 0.0
_integral   = 0.0

Traffic_Red = 0
Traffic_Green = 0
Slow_Sign = 0
Stop_Sign = 0
Left_Sign = 0
Right_Sign = 0
Straight_Sign = 0
Vehicle = 0


# ── 시각화 버퍼 (선택) ───────────────────────────────────────
error_buf = deque(maxlen=100)
steer_buf = deque(maxlen=100)

# ── 카메라 초기화 (전역 1회) ──────────────────────────────────
camera = CSICamera(capture_width=960,
                   capture_height=540,
                   capture_fps=30)
camera.running = True                              # JetCam 특성
# camera.cap.release()  
# ── BaseController 초기화 ───────────────────────────────────
base = BaseController('/dev/ttyUSB0', 115200)

# ── 프레임 획득 함수 ─────────────────────────────────────────
def get_frame():
    frame_rgb = camera.value                      # (H,W,3) RGB
    while frame_rgb is None:          # 시작 직후엔 None일 수 있음
        time.sleep(0.005)
        frame_rgb = camera.value

    # 이미지 크기 출력
    # print(f"[FRAME] RGB Frame shape: {frame_rgb.shape}")  # 예: (540, 960,)

    image_pil = Image.fromarray(frame_rgb)                 # PIL(RGB)
    image_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR) # OpenCV(BGR)
    h, w = frame_rgb.shape[:2]
    return image_pil, image_bgr, w, h

# ── Torch 모델 로드 및 전처리 ────────────────────────────────
device = torch.device('cuda')
model = torchvision.models.alexnet(num_classes=2, dropout=0.0)
model.load_state_dict(torch.load('./Lane_model/lane_best.pt',
                                 weights_only=True))
model = model.to(device).eval()

def preprocess(image_pil: Image.Image):
    return TEST_TRANSFORMS(image_pil).to(device)[None, ...]

# ── 제어 계산 함수들 ─────────────────────────────────────────
def pid_control(error: float) -> float:
    """정규화된 에러 → 스티어링 값(-1~1)"""
    global _prev_error, _integral
    dt = UPDATE_INTERVAL
    _integral   += error * dt
    derivative   = (error - _prev_error) / dt
    steer        = Kp*error + Ki*_integral + Kd*derivative
    _prev_error  = error
    return max(-1.0, min(steer, 1.0))

def compute_motor_output(steer: float, base_speed: float):
    """steer(+우, -좌) → L,R 속도"""
    turn_scale = MAX_STEER_RATIO * abs(steer)
    if steer < -0.15:          # 우회전 → 왼쪽 감속
        L = base_speed * (1.0 - turn_scale)
        R = base_speed * 2.2
    elif steer > 0.15:        # 좌회전 → 오른쪽 감속
        L = base_speed * 2.2
        R = base_speed * (1.0 - turn_scale)
    else:
        L = R = base_speed
    return L, R

def send_control(L: float, R: float):
    """BaseController 인터페이스 (로버는 -값이 전진)"""
    base.base_json_ctrl({"T": 11, "L": L, "R": R})

# ── 웨이포인트 추론 ──────────────────────────────────────────
def infer_waypoint():
    image_pil, image_bgr, width, height = get_frame()
    with torch.no_grad():
        output = model(preprocess(image_pil)).detach().cpu().numpy()[0]  # (x,y)
        x = (output[0] / 2 + 0.5) * width
        y = (output[1] / 2 + 0.5) * height
    # 시각화 (ESC 종료)
    # cv2.circle(image_bgr, (int(x), int(y)), 5, (0,0,255), -1)
    # image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    # cv2.imshow("Lane Prediction", image_rgb)
    if cv2.waitKey(1) == 27:
        raise KeyboardInterrupt
    return x, y, width

# ── 메인 제어 루프 ───────────────────────────────────────────
def run_pid_loop():
    try:
        while True:
            x, y, width = infer_waypoint()
            center_x = width / 2 - 50
            lateral_error = (x - center_x) / center_x    # [-1,1]
            steer = pid_control(lateral_error)

            # Task1: 신호등 인식
            if (Traffic_Red == 1):
                L, R = compute_motor_output(0.0, 0.0)
                send_control(L, R)
                print(f"[SIGNAL] RED LIGHT DETECTED! STOP!")
            elif (Traffic_Green == 1):
                L, R = compute_motor_output(steer, BASE_SPEED)
                send_control(L, R)
                print(f"[SIGNAL] GREEN LIGHT DETECTED! GO!")
            
            # Task2: 정지 Or 감속 사인 인식
            if (Slow_Sign == 1):
                L, R = compute_motor_output(steer, BASE_SPEED * 0.5)
                send_control(L, R)
                print(f"[SIGNAL] SLOW SIGN DETECTED! SLOW DOWN!")
            elif (Stop_Sign == 1):
                L, R = compute_motor_output(0.0, 0.0)
                send_control(L, R)
                print(f"[SIGNAL] STOP SIGN DETECTED! STOP!")
            
            # Task3: 회피주행

            # Task4: 직좌우 표지판 및 신호등 인식
            # if (Traffic_Red == 1):
            #     if 

            L, R  = compute_motor_output(steer, BASE_SPEED)
            send_control(L, R)

            error_buf.append(lateral_error)
            steer_buf.append(steer)
            print(f"[WAYPOINT] x={x:.1f}, error={lateral_error:+.2f}, center_x={center_x:.1f}"
                  f"steer={steer:+.2f} → L={L:.1f}, R={R:.1f}")
            time.sleep(UPDATE_INTERVAL)
    except KeyboardInterrupt:
        pass
    finally:
        send_control(0.0, 0.0)
        camera.running = False
        cv2.destroyAllWindows()

# ── (선택) 시각화 루프 ───────────────────────────────────────
# fig, ax = plt.subplots()
# ax.set_ylim(-1.5, 1.5); ax.set_xlim(0, 100)
# ln1, = ax.plot([], [], label='error'); ln2, = ax.plot([], [], label='steer')
# def update_plot(_):
#     ln1.set_data(range(len(error_buf)), error_buf)
#     ln2.set_data(range(len(steer_buf)), steer_buf)
#     return ln1, ln2
# ani = animation.FuncAnimation(fig, update_plot,
#                               interval=UPDATE_INTERVAL*1000, blit=True)
# plt.legend(); plt.show()

# ── 실행 엔트리포인트 ────────────────────────────────────────
if __name__ == "__main__":
    threading.Thread(target=run_pid_loop, daemon=True).start()
    while True:            # 메인 스레드 유지 (시각화 없을 때)
        time.sleep(1)
