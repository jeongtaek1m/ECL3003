# realtime_yolo_trt_save.py
from ultralytics import YOLO
import cv2, time, datetime, os
from pathlib import Path

# ========= 사용자 설정 =========
ENGINE_PATH     = "/home/ircv7/Embedded/Project_1/tensorrt_engine/250519_n_detection.engine"
SAVE_DIR        = Path("runs")               # 이미지·동영상 저장 폴더
SAVE_EVERY_N    = 10                        # N프레임마다 1장씩 이미지 저장
VIDEO_FPS       = 10                         # mp4 저장 FPS
# =================================

def gstreamer_pipeline(sensor_id=1, width=1920, height=1080,
                       display_width=960, display_height=540,
                       framerate=30, flip_method=0):
    return (
        "nvarguscamerasrc sensor-id=%d ! "
        "video/x-raw(memory:NVMM), width=%d, height=%d, framerate=%d/1 ! "
        "nvvidconv flip-method=%d ! "
        "video/x-raw, width=%d, height=%d, format=BGRx ! "
        "videoconvert ! "
        "video/x-raw, format=BGR ! appsink"
        % (sensor_id, width, height, framerate,
           flip_method, display_width, display_height)
    )

def main():

    
    # --- 초기화 ---
    model = YOLO(ENGINE_PATH)
    cap   = cv2.VideoCapture(gstreamer_pipeline(), cv2.CAP_GSTREAMER)
    if not cap.isOpened():
        raise RuntimeError("❌ 카메라 열기 실패")

    ok, frame = cap.read()
    if not ok:
        raise RuntimeError("첫 프레임을 읽을 수 없습니다.")

    print("▶ 추론 시작…  (Ctrl-C 로 중단)")

    frame_idx = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("⚠️  프레임 수신 실패 – 종료")
                break
            if frame_idx % SAVE_EVERY_N == 0:                    # N frame inference
                # --- YOLO 추론 ---
                result = model(frame, imgsz=640)[0]
                # print(result.boxes.xyxy)
                # print(result.boxes.conf)
                # print(result.boxes.cls)
            frame_idx += 1



    except KeyboardInterrupt:
        print("\n⏹️  사용자 중단")

    # --- 마무리 ---
    cap.release()

if __name__ == "__main__":
    main()
