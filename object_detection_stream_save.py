# realtime_yolo_trt_show.py
from ultralytics import YOLO
import cv2, time, datetime, os
from pathlib import Path

# ========= 사용자 설정 =========
ENGINE_PATH     = "/home/ircv7/Embedded/Project_1/tensorrt_engine/n_detection.engine"
PT_PATH         = "/home/ircv7/Embedded/Project_1/Object_detection/250519_n_detection.pt"
SAVE_DIR        = Path("runs")      # 이미지·동영상 저장 폴더
SAVE_EVERY_N    = 10               # N프레임마다 이미지 저장
VIDEO_FPS       = 10               # mp4 저장 FPS
WINDOW_NAME     = "YOLO-TRT Live"   # 실시간 창 이름
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
    model = YOLO(PT_PATH)
    cap   = cv2.VideoCapture(gstreamer_pipeline(), cv2.CAP_GSTREAMER)
    if not cap.isOpened():
        raise RuntimeError("❌ 카메라 열기 실패")

    SAVE_DIR.mkdir(parents=True, exist_ok=True)
    run_id   = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    img_dir  = SAVE_DIR / f"{run_id}_imgs"
    img_dir.mkdir()
    video_path = SAVE_DIR / f"{run_id}.mp4"

    ok, frame = cap.read()
    if not ok:
        raise RuntimeError("첫 프레임을 읽을 수 없습니다.")
    h, w = frame.shape[:2]
    vw = cv2.VideoWriter(str(video_path),
                         cv2.VideoWriter_fourcc(*"mp4v"),
                         VIDEO_FPS, (w, h))

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    print("▶ 추론 & 스트리밍 시작…  (ESC 또는 Ctrl-C 종료)")

    frame_idx = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("⚠️  프레임 수신 실패 – 종료")
                break

            # ── YOLO 추론 ───────────────────────────────────────────
            result      = model(frame, imgsz=640,conf=0.5)[0]
            vis_frame   = result.plot()   # BGR ndarray (box/라벨 포함)

            # ── 저장 ───────────────────────────────────────────────
            if frame_idx % SAVE_EVERY_N == 0:
                ts = datetime.datetime.now().strftime("%H%M%S_%f")
                cv2.imwrite(str(img_dir / f"{ts}.jpg"), vis_frame)
            vw.write(vis_frame)

            # ── 실시간 표시 ────────────────────────────────────────
            cv2.imshow(WINDOW_NAME, vis_frame)
            if cv2.waitKey(1) & 0xFF == 27:      # ESC
                print("⏹️  사용자 종료")
                break

            frame_idx += 1

    except KeyboardInterrupt:
        print("\n⏹️  사용자 중단")

    # --- 마무리 ---
    cap.release()
    vw.release()
    cv2.destroyAllWindows()
    print(f"✔️  동영상 저장: {video_path}")
    print(f"✔️  이미지 폴더: {img_dir}")

if __name__ == "__main__":
    main()
