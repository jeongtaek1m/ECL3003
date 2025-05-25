"""
Jetson-based 4WD 로버 — 모노 카메라 웨이포인트 추종 스크립트
──────────────────────────────────────────────────────────────
◎ 입력 : CSI 카메라 프레임
◎ 출력 : BaseController JSON 명령(L, R 좌우 바퀴 속도)
◎ 제어 : PID + 좌우 속도 차 감속 (MAX_STEER_RATIO)
"""

# ── 기본 라이브러리 ───────────────────────────────────────────
import time, math, threading, json
from collections import deque

import cv2
from PIL import Image
import numpy as np
import torch
import torchvision
# import matplotlib.pyplot as plt          # (시각화 주석 처리 가능)
# import matplotlib.animation as animation  # (시각화 주석 처리 가능)
import zmq

# ── Jetson / 프로젝트 라이브러리 ──────────────────────────────
from jetcam.csi_camera import CSICamera
from Lane_model.cnn.center_dataset import TEST_TRANSFORMS
from base_ctrl import BaseController

# ── 하이퍼파라미터 / 상수 ────────────────────────────────────
Kp, Ki, Kd       = 1.0, 0.05, 0.2               # PID gains

# if 255 then speed ; 1 then PWM
PWM_OR_SPEED = 255.0 * 2

UPDATE_INTERVAL  = 0.10                        # [s]
BASE_SPEED       = 70                          # 기본 속도 [%]
BASE_TURN_SPEED  = 100
MAX_STEER_RATIO  = 1.3                         # |steer|=1 일 때 감속비
OD_DEPTH_THRESH  = 8.0                         # OD로부터 받은 depth 기준

MAX_AVOID_SPEED = 150
MIN_AVOID_SPEED = -150
AVOID_TIME = 0.7

SLOW_CNT_THRESH = int(2.0 / UPDATE_INTERVAL)
STOP_CNT_THRESH = int(3.0 / UPDATE_INTERVAL)
slow_counter = 0
stop_counter = 0

# ── PID 상태 변수 ────────────────────────────────────────────
_prev_error = 0.0
_integral   = 0.0

Task1_Flag = 1
Task1_Red_Flag = 0

Task2_Flag = 0
time_gap_Flag = 0

Task3_Flag = 0
Task4_Flag = 0
Task4_Red_Flag = 0
Task4_Green1_Flag = 0
Task4_Green2_Flag = 0

# Slow_Flag = 0

LEFT, RIGHT, STRAIGHT = 0, 1, 2
SLOW, STOP           = 3, 4
GREEN, RED           = 5, 6
VEHICLE              = 7

ALL_CLASSES = [LEFT, RIGHT, STRAIGHT,
               SLOW, STOP,
               GREEN, RED,
               VEHICLE]

history = deque(maxlen=5)

# # ── 시각화 버퍼 (선택) ───────────────────────────────────────
# error_buf = deque(maxlen=100)
# steer_buf = deque(maxlen=100)

# ── 카메라 초기화 (전역 1회) ──────────────────────────────────
camera = CSICamera(capture_width=960,
                   capture_height=540,
                   capture_fps=30)
camera.running = True                              # JetCam 특성
# camera.cap.release()  
# ── BaseController 초기화 ───────────────────────────────────
base = BaseController('/dev/ttyUSB0', 115200)

# ── ZMQ SUB 초기화 ───────────────────────────────────────────
ctx = zmq.Context()
sub = ctx.socket(zmq.SUB)
sub.connect("tcp://localhost:5555")  # OD 쪽 PUB 주소
sub.setsockopt_string(zmq.SUBSCRIBE, "")
latest_frame = []

_last_flag_print_time = 0.0

def zmq_listener():
    global latest_frame
    while True:
        msg = sub.recv_string()
        data = json.loads(msg)
        latest_frame = data.get("detections", [])

def update_history(detections: list[dict]):

    history.append(detections)

def compute_depths(threshold):

    global _last_flag_print_time

    depths = {}
   
    for cls in ALL_CLASSES:
        # 모든 프레임에서 해당 클래스의 depth 값들만 뽑아 내기
        vals = [
            det["depth"]
            for frame in history
            for det   in frame
            if det["class"] == cls
        ]
        if len(vals) >= threshold:
            depths[cls] = sum(vals) / len(vals)
        else:
            depths[cls] = 0

    now = time.time()
    if now - _last_flag_print_time >= 0.5:
        active = [f"{cls}:{depths[cls]:.2f}" for cls in ALL_CLASSES if depths[cls] is not None]
        if active:
            print(f"[DEBUG] Avg depths (count>={threshold}): " + ", ".join(active))
        else:
            print("[DEBUG] No classes meet depth threshold")
        _last_flag_print_time = now

    return depths
    #flags = {}
    # counts = {}

    # for cls in ALL_CLASSES:
    #     cnt = sum(1 for fr in history if cls in fr)
    #     counts[cls] = cnt
    #     flags[cls] = (cnt >= threshold)

    # now = time.time()
    # if now -_last_flag_print_time >= 1.0:    
    #     print(f"[DEBUG] history={list(history)}")
    #     print(f"[DEBUG] counts={counts}")
    #     _last_flag_print_time = now

    # return flags
    

threading.Thread(target=zmq_listener, daemon=True).start()

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
model = torchvision.models.alexnet(num_classes=2)
model.load_state_dict(torch.load('./Lane_model/lane_best_521.pt',
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
        # base_speed = BASE_TURN_SPEED
        L = base_speed * (1.0 - turn_scale)
        R = base_speed * 2.5
    elif steer > 0.15:        # 좌회전 → 오른쪽 감속
        # base_speed = BASE_TURN_SPEED
        L = base_speed * 2.5
        R = base_speed * (1.0 - turn_scale)
    else:
        L = R = base_speed
    return L, R

def send_control(L: float, R: float):
    """BaseController 인터페이스 (로버는 -값이 전진)"""
    L = L / PWM_OR_SPEED
    R = R / PWM_OR_SPEED
    base.base_json_ctrl({"T": 1, "L": L, "R": R})

# ── 웨이포인트 추론 ──────────────────────────────────────────
def infer_waypoint():
    image_pil, image_bgr, width, height = get_frame()
    with torch.no_grad():
        output = model(preprocess(image_pil)).detach().cpu().numpy()[0]  # (x,y)
        x = (output[0] / 2 + 0.5) * width
        y = (output[1] / 2 + 0.5) * height
    # 시각화 (ESC 종료)
    cv2.circle(image_bgr, (int(x), int(y)), 5, (0,0,255), -1)
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    cv2.imshow("Lane Prediction", image_rgb)
    if cv2.waitKey(1) == 27:
        raise KeyboardInterrupt
    return x, y, width

# def infer_object():
#     image_pil, image_bgr, width, height = get_frame()
#     with torch.no_grad():
#         result = model(image_pil)
#         box_area = abs((result.boxes.xyxy[3] - result.boxes.xyxy[1]) * (result.boxes.xyxy[2] - result.boxes.xyxy[0]))
#         box_conf = result.boxes.conf
#         box_cls = result.boxes.cls
#         # print(result.boxes.xyxy)
#         # print(result.boxes.conf)
#         # print(result.boxes.cls)
#     return box_area, box_conf, box_cls

# ── 메인 제어 루프 ───────────────────────────────────────────
# global time_gap
# time_gap = 0
def run_pid_loop():
    global latest_frame
    global Task1_Flag
    global Task2_Flag
    global Task3_Flag
    global Task4_Flag
    global Task1_Red_Flag
    global Task4_Red_Flag
    global Task4_Green1_Flag
    global Task4_Green2_Flag
    # global Slow_Flag
    global time_gap_Flag
    global slow_counter, stop_counter

    TASK_SPEED = 50

    try:
        while True:
            x, y, width = infer_waypoint()
            center_x = width / 2 - 50
            lateral_error = (x - center_x) / center_x    # [-1,1]
            steer = pid_control(lateral_error)
            # TASK_SPEED = 100

            # 2) OD→Depth 정보 처리
            # if latest_frame:
            #     for det in latest_frame:
            #         cls   = det["class"]   # 정수
            #         cx    = det["cx"]      # 정수
            #         cy    = det["cy"]      # 정수
            #         depth = det["depth"]   # float

            #         # 예시 출력
            #         print(f"Object class={cls}, center=({cx},{cy}), depth={depth:.2f}")

                # classes = [det["class"] for det in latest_frame]   # [0, 1, …]
                # cxs     = [det["cx"]    for det in latest_frame]   # [123, 345, …]
                # cys     = [det["cy"]    for det in latest_frame]   # [234, 456, …]
                # depths  = [det["depth"] for det in latest_frame]   # [2.5, 1.2, …]

                # depths = [d["depth"] for d in latest_frame]
                #min_depth = max(depths)
               
                # 예시: 너무 가까우면 정지
                # if min_depth > OD_DEPTH_THRESH:
                #     print(f"[OD] class={cls} too close (depth={min_depth:.2f}), STOP")
                #     send_control(0,0)
                #     time.sleep(1.0)

                # 사용 후 초기화
                #latest_frame = []

            # detections = [det["class"] for det in latest_frame]
            # update_history(detections)
            # flags = compute_flags(threshold=2)

            update_history(latest_frame)
            depths = compute_depths(threshold = 2)

            vehicle_cx = [det["cx"] for det in latest_frame if det["class"] == VEHICLE]
            if vehicle_cx:
                vehicle_center = sum(vehicle_cx) / len(vehicle_cx)
                print(f"[DEBUG] Vehicle cx list = {vehicle_cx}, avg = {vehicle_center}")
            else:
                vehicle_center = None  # Vehicle 검출 없을 때

            latest_frame = []

            if (Task1_Flag):  
                # Task1: 신호등 인식
                print(f"Task1 Start!!")
                if (depths[RED] is not None and depths[RED] > 2):

                    TASK_SPEED = 0
                    print(f"[SIGNAL] RED LIGHT DETECTED! STOP!")
                    Task1_Red_Flag = 1
                   
                elif (depths[GREEN] > 0 and Task1_Red_Flag == 1):
                    TASK_SPEED = BASE_SPEED
                    print(f"[SIGNAL] GREEN LIGHT DETECTED! GO!")
                    if (Task1_Red_Flag == 1):
                       Task1_Flag = 0
                       Task2_Flag = 1
                       Task1_Red_Flag = 0
            

            # comment : 그냥 depth trigger 걸렸을 때 시간 재고 해당 몇초 지나면 slow 랑 stop 알아서 풀리게끔
            elif (Task2_Flag):
                # Task2: 정지 Or 감속 사인 인식
                print(f"Task2 Start!!")
                if (depths[SLOW] > 1):
                    TASK_SPEED = 50
                    print(f"[SIGNAL] SLOW SIGN DETECTED! SLOW DOWN!")

                    slow_counter += 1
                    stop_counter = 0
                    time_gap_Flag = 1

                    # 2초(=20프레임) 연속 검출되면 다음 태스크로
                    if slow_counter >= SLOW_CNT_THRESH:
                        Task2_Flag = 0
                        Task3_Flag = 1
                        TASK_SPEED = BASE_SPEED
                        slow_counter = 0


                    # if (time_gap_Flag == 0):
                    #     prev_time = time.time()
                    #     time_gap_Flag = 1
                    # time_gap = time.time() - prev_time
                    # if (time_gap > 2):
                    #     Task2_Flag = 0
                    #     Task3_Flag = 1
                    #     TASK_SPEED = 100

                    # Slow_Flag = 1
                
                # elif (depths[SLOW] is None and Slow_Flag == 1):
                #     TASK_SPEED = 100
                #     Task2_Flag = 0
                #     Task3_Flag = 1

                elif (depths[STOP] > 1):
                    TASK_SPEED = 0
                    print(f"[SIGNAL] STOP SIGN DETECTED! STOP!")

                    stop_counter += 1
                    slow_counter = 0
                    time_gap_Flag = 1

                    # 3초(=30프레임) 연속 검출되면 다음 태스크로
                    if stop_counter >= STOP_CNT_THRESH:
                        Task2_Flag = 0
                        Task3_Flag = 1
                        TASK_SPEED = BASE_SPEED
                        stop_counter = 0
                    # if (time_gap_Flag == 0):
                    #     prev_time = time.time()
                    #     time_gap_Flag = 1
                    # time_gap = time.time() - prev_time
                    # if (time_gap > 3):
                    #     Task2_Flag = 0
                    #     Task3_Flag = 1
                    #     TASK_SPEED = 100

                # comment : 이거 필요없는거 아닌가?
                elif (time_gap_Flag):
                    slow_counter += 1
                    stop_counter += 1
                    
                    if slow_counter >= SLOW_CNT_THRESH or stop_counter >= STOP_CNT_THRESH:
                        Task2_Flag = 0
                        Task3_Flag = 1
                        TASK_SPEED = BASE_SPEED
                        slow_counter = 0
                        stop_counter = 0

            elif (Task3_Flag):
                # Task3: 회피주행
                # comment : 사다리꼴
                print(f"Task3 Start!!")
                if (depths[VEHICLE] is not None and depths[VEHICLE] > 6):
                    if (vehicle_center is not None and vehicle_center < center_x):  # 오른쪽으로 회피
                        send_control(MAX_AVOID_SPEED,MIN_AVOID_SPEED) # 오른쪽으로 2초 이동
                        time.sleep(AVOID_TIME)
                        send_control(150,150)
                        time.sleep(1.0)
                        send_control(MIN_AVOID_SPEED,MAX_AVOID_SPEED) # 왼쪽으로 2초 이동
                        time.sleep(AVOID_TIME)
                        send_control(150,150) # 직진 1초 이동
                        time.sleep(1.0)
                        send_control(MIN_AVOID_SPEED,MAX_AVOID_SPEED) # 왼쪽으로 2초 이동
                        time.sleep(AVOID_TIME)
                        # send_control(150,150)
                        # time.sleep(1.0)
                        # send_control(MAX_AVOID_SPEED, MIN_AVOID_SPEED)
                        # time.sleep(AVOID_TIME)
                        # send_control(MAX_AVOID_SPEED,MIN_AVOID_SPEED) # 오른쪽으로 2초 이동
                        # time.sleep(1.5)
                        
                    elif (vehicle_center is not None and vehicle_center > center_x):  # 왼쪽으로 회피
                        send_control(MIN_AVOID_SPEED,MAX_AVOID_SPEED) # 왼쪽으로 2초 이동
                        time.sleep(AVOID_TIME)
                        send_control(150,150)
                        time.sleep(1.0)
                        send_control(MAX_AVOID_SPEED,MIN_AVOID_SPEED) # 오른쪽으로 2초 이동
                        time.sleep(AVOID_TIME)
                        send_control(150,150) # 직진 1초 이동
                        time.sleep(1.0)
                        send_control(MAX_AVOID_SPEED,MIN_AVOID_SPEED) # 오른쪽으로 2초 이동
                        time.sleep(AVOID_TIME)
                        # send_control(150,150)
                        # time.sleep(1.0)
                        # send_control(MIN_AVOID_SPEED, MAX_AVOID_SPEED)
                        # time.sleep(AVOID_TIME)
                        # send_control(MIN_AVOID_SPEED,MAX_AVOID_SPEED) # 왼쪽으로 2초 이동
                        # time.sleep(1.5)

                    Task3_Flag = 0
                    Task4_Flag = 1

            elif (Task4_Flag):
                # Task4: 직좌우 표지판 및 신호등 인식
                print(f"Task4 Start!!")
                #TASK_SPEED = 70

                if depths[RED] > depths[LEFT] + depths[RIGHT] + depths[STRAIGHT]:
                    TASK_SPEED = 0
                    print(f"[SIGNAL] RED LIGHT DETECTED! STOP!")
                elif depths[GREEN] > depths[LEFT] + depths[RIGHT] + depths[STRAIGHT]:
                    TASK_SPEED = BASE_SPEED
                
                if depths[LEFT] > 4 and TASK_SPEED != 0:
                    send_control(-10,200)
                    time.sleep(3)
                elif depths[RIGHT] > 4 and TASK_SPEED != 0:
                    send_control(200,-10)
                    time.sleep(3)
                elif depths[STRAIGHT] > 4 and TASK_SPEED != 0:
                    TASK_SPEED = BASE_SPEED
                
                # if depths[RED] > depths[LEFT] + depths[RIGHT] + depths[STRAIGHT] and depths[RED] > 0: # 빨간불인 경우 정지
                #     TASK_SPEED = 0
                #     print(f"[SIGNAL] RED LIGHT DETECTED! STOP!")
                #     Task4_Red_Flag = 1

                # elif Task4_Green1_Flag and Task4_Red_Flag:
                #     if depths[GREEN] > depths[LEFT] + depths[RIGHT] + depths[STRAIGHT]:
                #         Task4_Green2_Flag = 1

                #     if Task4_Green2_Flag:
                #         if (depths[LEFT] is not None and depths[LEFT] > 3):
                #             send_control(-10,200)
                #             time.sleep(3)
                #         elif (depths[RIGHT] is not None and depths[RIGHT] > 3) :
                #             send_control(200,-10)
                #             time.sleep(3)
                #         elif (depths[STRAIGHT] is not None and depths[STRAIGHT] > 3):
                #             TASK_SPEED = BASE_SPEED 
                #         else:
                #             TASK_SPEED = BASE_SPEED
                    
                # elif depths[GREEN] > depths[LEFT] + depths[RIGHT] + depths[STRAIGHT] and depths[GREEN] > 2: 
                #     Task4_Green1_Flag = 1
                #     if Task4_Red_Flag == 0:
                #         # Task4_Green_Flag = 1
                #         TASK_SPEED = 40
                #     else:
                #         TASK_SPEED = BASE_SPEED  
                                                     # 빨간불이 아니고 초록불인 경우  
                         
                        
                # elif Task4_Red_Flag:
                #     TASK_SPEED = 0
            
            speed = min(TASK_SPEED, BASE_SPEED)
            # if abs(steer) > 0.15:
            #     speed = BASE_TURN_SPEED 
            L, R  = compute_motor_output(steer, speed)
            send_control(L, R)

            # if Task1_Flag:
            #     send_control(TASK_SPEED, TASK_SPEED)

            # error_buf.append(lateral_error)
            # steer_buf.append(steer)
            # print(f"[WAYPOINT] x={x:.1f}, error={lateral_error:+.2f}, center_x={center_x:.1f}"
            #       f"steer={steer:+.2f} → L={L:.1f}, R={R:.1f}")
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
