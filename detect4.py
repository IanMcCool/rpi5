import threading
import time
import cv2
import serial
import numpy as np
from ultralytics import YOLO
from picamera2 import Picamera2

# ─── CONFIGURATION ────────────────────────────────────────────────────────────
RESOLUTION   = (640, 480)
TARGET_FPS   = 60

FRAME_WIDTH, FRAME_HEIGHT = RESOLUTION
DEAD_FRAC    = 0.30  # central 30% dead‑zone
DEAD_X       = int(FRAME_WIDTH  * DEAD_FRAC * 0.5)  # ±15% of width
DEAD_Y       = int(FRAME_HEIGHT * DEAD_FRAC * 0.5)  # ±15% of height

MODEL_PATH   = "/home/XenaPi/yolo/yolov8n_ncnn_model"
THRESHOLD    = 0.5
# ────────────────────────────────────────────────────────────────────────────────

# Set up serial
try:
    serial_port = serial.Serial(port='/dev/ttyACM0', baudrate=9600, timeout=1)
    print("Serial port opened successfully.")
except Exception as e:
    print("Failed to open serial port:", e)
    serial_port = None

class PiCameraStream:
    def __init__(self, resolution=RESOLUTION, target_fps=TARGET_FPS):
        self.camera = Picamera2()
        config = self.camera.create_video_configuration(
            main={"format": "XRGB8888", "size": resolution}
        )
        self.camera.configure(config)
        self.camera.set_controls({"FrameRate": target_fps})
        self.camera.start()
        self.frame = None
        self.stopped = False
        self.lock = threading.Lock()

    def update(self):
        while not self.stopped:
            frame_bgra = self.camera.capture_array()
            frame = cv2.cvtColor(frame_bgra, cv2.COLOR_BGRA2BGR)
            with self.lock:
                self.frame = frame

    def read(self):
        with self.lock:
            return self.frame.copy() if self.frame is not None else None

    def stop(self):
        self.stopped = True
        self.camera.stop()

def center_bound(xmin, ymin, xmax, ymax, frame_x, frame_y):
    box_x = (xmin + xmax) / 2
    box_y = (ymin + ymax) / 2
    frame_x_center = frame_x / 2
    frame_y_center = frame_y / 2
    offset_x = frame_x_center - box_x
    offset_y = frame_y_center - box_y
    return int(offset_x), int(offset_y)

def send_and_wait_for_echo(offset_x, offset_y):
    expected = f"({offset_x},{offset_y})"
    msg = expected + "\r\n"
    if not (serial_port and serial_port.is_open):
        print("No serial port; would send:", expected)
        return
    serial_port.reset_input_buffer()
    serial_port.write(msg.encode('ascii'))
    serial_port.flush()
    print("Sent:", expected)

    # wait for exact echo
    while True:
        line = serial_port.readline()
        if not line:
            continue
        echo = line.decode('ascii', errors='ignore').strip()
        print("Echo:", echo)
        if echo == expected:
            break

def inference_loop(get_frame_func, model, labels, threshold, bbox_colors):
    print(f"Dead‑zone = ±({DEAD_X}, {DEAD_Y}) pixels")
    while True:
        frame = get_frame_func()
        if frame is None:
            continue

        results = model(frame, verbose=False)
        detections = results[0].boxes

        for det in detections:
            xmin, ymin, xmax, ymax = det.xyxy.cpu().numpy().squeeze().astype(int)
            cls_idx = int(det.cls.item())
            classname = labels[cls_idx]
            conf = det.conf.item()

            if conf < threshold:
                continue

            color = bbox_colors[cls_idx % len(bbox_colors)]
            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), color, 2)
            label = f"{classname}: {int(conf*100)}%"
            cv2.putText(frame, label, (xmin, ymin - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

            if classname.lower() == 'person':
                h, w = frame.shape[:2]
                offset_x, offset_y = center_bound(xmin, ymin, xmax, ymax, w, h)
                cx = int((xmin + xmax) / 2)
                cy = int((ymin + ymax) / 2)
                cv2.circle(frame, (cx, cy), 5, (0, 255, 0), -1)

                # only send when outside dead‑zone
                if abs(offset_x) > DEAD_X or abs(offset_y) > DEAD_Y:
                    print(f"Offset ({offset_x},{offset_y}) outside DZ")
                    send_and_wait_for_echo(offset_x, offset_y)
                else:
                    print(f"Within dead‑zone ±({DEAD_X},{DEAD_Y}); no send")

        cv2.imshow('YOLO detection results', frame)
        if cv2.waitKey(1) in (ord('q'), ord('Q')):
            break

def main():
    # camera capture thread
    stream = PiCameraStream()
    cap_thread = threading.Thread(target=stream.update, daemon=True)
    cap_thread.start()

    # load model
    model = YOLO(MODEL_PATH, task='detect')
    labels = model.names
    bbox_colors = [
        (164, 120, 87), (68, 148, 228), (93, 97, 209),
        (178, 182, 133), (88, 159, 106), (96, 202, 231),
        (159, 124, 168), (169, 162, 241), (98, 118, 150),
        (172, 176, 184)
    ]

    inf_thread = threading.Thread(
        target=inference_loop,
        args=(stream.read, model, labels, THRESHOLD, bbox_colors),
        daemon=True
    )
    inf_thread.start()

    try:
        while inf_thread.is_alive():
            time.sleep(0.1)
    except KeyboardInterrupt:
        pass

    stream.stop()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
