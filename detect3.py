import threading
import time
import cv2
import serial
import numpy as np
from ultralytics import YOLO
from picamera2 import Picamera2

uart = serial.Serial('/dev/ttyAMA0', baudrate=115200, timeout=1)

class PiCameraStream:
    def __init__(self, resolution=(640, 480), target_fps=60):
        self.camera = Picamera2()
        config = self.camera.create_video_configuration(
            main={"format": "XRGB8888", "size": resolution}
        )
        self.camera.configure(config)
        # Request a higher frame rate (ensure the sensor mode supports this)
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
    box_x = (xmin + xmax)/2
    box_y = (ymin + ymax)/2
    
    #creating a var for frame allows us to get center and change res
    frame_x_center = frame_x/2
    frame_y_center = frame_y/2
        
    offset_x = frame_x - box_x
    offset_y = frame_y - box_y
    
    #return and caste as int for pwm values
    return float(offset_x), float(offset_y)
        
        
        

def inference_loop(get_frame_func, model, labels, threshold, bbox_colors):
    while True:
        frame = get_frame_func()
        if frame is None:
            continue
        # Run inference
        results = model(frame, verbose=False)
        detections = results[0].boxes
        # Draw detections
        for detection in detections:
            xyxy_tensor = detection.xyxy.cpu()
            xyxy = xyxy_tensor.numpy().squeeze()  # [xmin, ymin, xmax, ymax]
            xmin, ymin, xmax, ymax = xyxy.astype(int)
            #print("offset x:", {my_offset_x}, "\n offset y:", {my_offset_y})
            classidx = int(detection.cls.item())
            classname = labels[classidx]
            conf = detection.conf.item()
            if conf > threshold:
                color = bbox_colors[classidx % len(bbox_colors)]
                cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), color, 2)
                label = f'{classname}: {int(conf * 100)}%'
                labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                label_ymin = max(ymin, labelSize[1] + 10)
                cv2.rectangle(frame, (xmin, label_ymin - labelSize[1] - 10),
                              (xmin + labelSize[0], label_ymin + baseLine - 10), color, cv2.FILLED)
                cv2.putText(frame, label, (xmin, label_ymin - 7),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
                
                if classname.lower() == 'person':
                    frame_height, frame_width = frame.shape[:2]
                    offset_x, offset_y = center_bound(xmin, ymin, xmax, ymax, frame_width, frame_height)
                    
                    center_x = float((xmin + xmax) / 2)
                    center_y = float((ymin + ymax) / 2)
                    
                    cv2.circle(frame, (center_x, center_y), 5, (0, 255, 0), -1)
                    print(f"Person detected. Offset (x, y): ({offset_x}, {offset_y})")
                    
                    message = f"{offset_x},{offset_y}\n"
                    uart.write(message.encode())
                
        cv2.imshow('YOLO detection results', frame)
        if cv2.waitKey(1) in [ord('q'), ord('Q')]:
            break

def main():
    # Set up the PiCamera stream with high target FPS
    stream = PiCameraStream(resolution=(640, 480), target_fps=60)
    capture_thread = threading.Thread(target=stream.update, daemon=True)
    capture_thread.start()

    # Load YOLO model; adjust the model path as needed
    model_path = "/home/XenaPi/CV2_PRO/yolov8n_ncnn_model"
    model = YOLO(model_path, task='detect')
    labels = model.names
    threshold = 0.5
    bbox_colors = [
        (164, 120, 87), (68, 148, 228), (93, 97, 209), (178, 182, 133), (88, 159, 106),
        (96, 202, 231), (159, 124, 168), (169, 162, 241), (98, 118, 150), (172, 176, 184)
    ]

    # Start inference in a dedicated thread so capture and inference run concurrently.
    inference_thread = threading.Thread(
        target=inference_loop,
        args=(stream.read, model, labels, threshold, bbox_colors),
        daemon=True
    )
    inference_thread.start()

    # Keep the main thread alive until user quits
    try:
        while inference_thread.is_alive():
            time.sleep(0.1)
    except KeyboardInterrupt:
        pass

    stream.stop()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
