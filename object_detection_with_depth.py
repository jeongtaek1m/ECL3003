# realtime_yolo_trt_save.py
from ultralytics import YOLO
import cv2, time, datetime, os
from pathlib import Path
from Depth_Anything_for_Jetson_Orin.depth_with_yolo import DepthEngine

import zmq
import json


# ========= 사용자 설정 =========
ENGINE_PATH     = "/home/ircv7/Embedded/Project_1/tensorrt_engine/250521_n_detection.engine"
SAVE_DIR        = Path("runs")               # 이미지·동영상 저장 폴더
SAVE_VIDEO      = False                       # 영상 저장 활성화
VIDEO_NAME      = "yolo_depth_output.mp4"    # 저장될 비디오 파일명
VIDEO_FPS       = 5                          # 비디오 저장 FPS
# =================================


def gstreamer_pipeline(sensor_id=1, width=960, height=540,
                       display_width=960, display_height=540,
                       framerate=30, flip_method=0):
    return (
        "nvarguscamerasrc sensor-id=%d ! "
        "video/x-raw(memory:NVMM), width=(int)%d, height=(int)%d, format=(string)NV12, framerate=(fraction)%d/1 ! "
        "nvvidconv ! "
        "video/x-raw, format=(string)BGRx ! "
        "videoconvert ! "
        "video/x-raw, format=(string)BGR ! "
        "appsink max-buffers=1 drop=True"
        % (
            sensor_id,
            width,
            height,
            framerate,
        )
    )


def main():
    # --- ZMQ PUB 초기화 === ADDED ===
    ctx   = zmq.Context()
    pub   = ctx.socket(zmq.PUB)
    pub.bind("tcp://*:5555")  # 원하는 포트로 변경 가능
    # ========================

    # --- 저장을 위한 VideoWriter 초기화 ---
    video_writer = None
    if SAVE_VIDEO:
        SAVE_DIR.mkdir(parents=True, exist_ok=True)
    
    # --- 초기화 ---
    cap = cv2.VideoCapture(gstreamer_pipeline(), cv2.CAP_GSTREAMER)
    if not cap.isOpened():
        raise RuntimeError("❌ 카메라 열기 실패")

    ok, frame = cap.read()
    if not ok:
        raise RuntimeError("첫 프레임을 읽을 수 없습니다.")

    # VideoWriter 파라미터 설정
    if SAVE_VIDEO:
        height, width = frame.shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out_path = SAVE_DIR / VIDEO_NAME
        video_writer = cv2.VideoWriter(str(out_path), fourcc, VIDEO_FPS, (width, height))
        print(f"🎥 Saving video to {out_path}")

    print("============INFERENCE START==================")
    depth = DepthEngine(
        frame_rate=10,
        raw=True,
        stream=False, 
        record=False,
        save=False, 
        grayscale=False
    )
    # depth._width, depth._height = 1280, 720
    
    yolo_model = YOLO(ENGINE_PATH)
    frame_idx = 0

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("⚠️  프레임 수신 실패 – 종료")
                break

            if frame_idx % 5 == 0:                    # N frame inference
                # --- YOLO 추론 ---
                bbox_result = yolo_model(frame, imgsz=640, conf=0.45)[0]
                boxes = bbox_result.boxes.xyxy.cpu().numpy()
                classes = bbox_result.boxes.cls.cpu().numpy()
                depth_map = depth.infer(frame)

                msgs = []
                for i, (x1, y1, x2, y2) in enumerate(boxes):
                    cx = int((x1 + x2) / 2)
                    cy = int((y1 + y2) / 2)
                    depth_val = float(depth_map[cy, cx].item())
                    print(f"[Object {i}] center=({cx},{cy}) → depth = {depth_val:.3f}")

                    msgs.append({
                        "class": int(classes[i]),
                        "cx":    cx,
                        "cy":    cy,
                        "depth": depth_val
                    })

                # --- 클래스별로 하나만 선택 (예: Depth가 작은 것 우선) ---
                best_dets = {}
                for det in msgs:
                    cls = det["class"]
                    # 처음 들어오거나, 더 작은 depth 면 교체
                    if cls not in best_dets or det["depth"] > best_dets[cls]["depth"]:
                        best_dets[cls] = det

                # 프레임 단위로 한 번에
                final_msgs = list(best_dets.values())
                frame_msg = {
                    "time":       time.time(),
                    "detections": final_msgs
                }
                pub.send_string(json.dumps(frame_msg))

                # (선택) 시각화 및 저장
                vis = bbox_result.plot()
                cv2.imshow("YOLO+Depth", vis)

                if SAVE_VIDEO and video_writer is not None:
                    video_writer.write(vis)

            frame_idx += 1
            if cv2.waitKey(1) == ord('q'):
                break

    except KeyboardInterrupt:
        print("\n⏹️  사용자 중단")

    # --- 마무리 ---
    cap.release()
    if video_writer is not None:
        video_writer.release()
        print("🎬 Video saved successfully.")

if __name__ == "__main__":
    main()
