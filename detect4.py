import threading
import time
import cv2
import serial
import numpy as np
from ultralytics import YOLO
from picamera2 import Picamera2

def send_coords(ser, x, y):
    """
    Send integer x,y over already-open serial port as "x,y\r\n".
    """
    cx = int(x)
    cy = int(y)
    msg = f"{cx},{cy}\r\n".encode("utf-8")
    ser.write(msg)

class PiCameraStream:
    def __init__(self, resolution=(640, 480), target_fps=60):
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
    offset_x = frame_x - box_x
    offset_y = frame_y - box_y
    return offset_x, offset_y
        
def inference_loop(get_frame, model, labels, threshold, bbox_colors, ser):
    while True:
        frame = get_frame()
        if frame is None:
            continue

        results = model(frame, verbose=False)
        detections = results[0].boxes

        for det in detections:
            xmin, ymin, xmax, ymax = det.xyxy.cpu().numpy().squeeze().astype(int)
            conf = det.conf.item()
            cls = int(det.cls.item())
            name = labels[cls]

            if conf > threshold:
                # draw box + label
                color = bbox_colors[cls % len(bbox_colors)]
                cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), color, 2)
                label = f"{name}: {int(conf*100)}%"
                (w, h), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                y0 = max(ymin, h + 10)
                cv2.rectangle(frame, (xmin, y0-h-10), (xmin+w, y0+baseline-10), color, cv2.FILLED)
                cv2.putText(frame, label, (xmin, y0-7), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1)

                if name.lower() == "person":
                    fh, fw = frame.shape[:2]
                    offset_x, offset_y = center_bound(xmin, ymin, xmax, ymax, fw, fh)
                    center_x = (xmin + xmax) / 2
                    center_y = (ymin + ymax) / 2

                    # draw center point
                    cv2.circle(frame, (int(center_x), int(center_y)), 5, (0,255,0), -1)

                    # send coords over UART to Nucleo
                    send_coords(ser, center_x, center_y)

        cv2.imshow("YOLO detection results", frame)
        if cv2.waitKey(1) in (ord('q'), ord('Q')):
            break

def main():
    # 1) Open the Nucleo's USB‑serial (likely /dev/ttyACM0) at 9600 baud:
    ser = serial.Serial("/dev/ttyACM0", baudrate=9600, timeout=1)

    # 2) Start the camera thread
    stream = PiCameraStream(resolution=(640, 480), target_fps=60)
    t_cam = threading.Thread(target=stream.update, daemon=True)
    t_cam.start()

    # 3) Load the YOLO model
    model = YOLO("/home/XenaPi/yolo/yolov8n_ncnn_model", task="detect")
    labels = model.names
    threshold = 0.5
    bbox_colors = [
        (164,120,87), (68,148,228), (93,97,209), (178,182,133), (88,159,106),
        (96,202,231), (159,124,168), (169,162,241), (98,118,150), (172,176,184)
    ]

    # 4) Launch inference, passing the open serial port
    t_inf = threading.Thread(
        target=inference_loop,
        args=(stream.read, model, labels, threshold, bbox_colors, ser),
        daemon=True
    )
    t_inf.start()

    # 5) Keep alive until user quits
    try:
        while t_inf.is_alive():
            time.sleep(0.1)
    except KeyboardInterrupt:
        pass

    # 6) Clean up
    stream.stop()
    ser.close()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
