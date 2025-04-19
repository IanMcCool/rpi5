import threading
import time
import cv2
import serial
import math
import numpy as np
from ultralytics import YOLO
from picamera2 import Picamera2


try:
    serial_port = serial.Serial(port='/dev/ttyACM0', baudrate=9600, timeout=1)
    print("Serial port opened successfully.")
except Exception as e:
    print("Failed to open serial port:", e)
    serial_port = None
    
    
V_FOV = 174
H_FOV = 172

M_PAN = 2.0 * 3.67816
M_TILT = 2.0 * 2.85714


leftPulse   = 0.05;   
centerPulse = 0.075;   
rightPulse  = 0.10;     
step = 0.001;        
delay_ms = 20;

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
        
    offset_x = frame_x_center - box_x
    
    offset_y = box_y - frame_y_center
    
    #return and cast as int for pwm values
    return int(offset_x), int(offset_y)
    
def calculate_servo_angle(offset_x, offset_y):
    # 1) pixel?degree
    delta_pan  = offset_x  / M_PAN    # in degrees
    delta_tilt = offset_y  / M_TILT   # in degrees

    # 2) add home position
    angle_pan  = 90.0 + delta_pan
    angle_tilt = 90.0 + delta_tilt

   
    angle_pan  = max(0.0, min(180.0, angle_pan))
    angle_tilt = max(0.0, min(180.0, angle_tilt))

    span = rightPulse - leftPulse
    duty_pan  = leftPulse + (angle_pan  / 180.0) * span
    duty_tilt = leftPulse + (angle_tilt / 180.0) * span

    return duty_pan, duty_tilt
    

    
def send_and_wait_for_echo(duty_pan, duty_tilt):
    # round both values
   

    # Format the values with exactly 3 decimal places
    msg = f"{duty_pan:.3f} {duty_tilt:.3f}\r\n"

    if not (serial_port and serial_port.is_open):
        print("No serial port; would send:", msg.strip())
        return

    # clear old input, send and flush
    serial_port.reset_input_buffer()
    serial_port.write(msg.encode('ascii'))
    serial_port.flush()
    print(f"Sent: Pan={duty_pan:.3f}, Tilt={duty_tilt:.3f}")

   
    

        
        

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
                 if classname.lower() == 'bottle':
                    color = bbox_colors[classidx % len(bbox_colors)]
                    cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), color, 2)
                    label = f'{classname}: {int(conf * 100)}%'
                    labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                    label_ymin = max(ymin, labelSize[1] + 10)
                    cv2.rectangle(frame, (xmin, label_ymin - labelSize[1] - 10),
                              (xmin + labelSize[0], label_ymin + baseLine - 10), color, cv2.FILLED)
                    cv2.putText(frame, label, (xmin, label_ymin - 7),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
                
               
                    frame_height, frame_width = frame.shape[:2]
                    offset_x, offset_y = center_bound(xmin, ymin, xmax, ymax, frame_width, frame_height)
                    
                    duty_pan, duty_tilt = calculate_servo_angle(offset_x, offset_y)
                   #print(f"Person detected. Offset (x, y): ({offset_x}, {offset_y})")
                    send_and_wait_for_echo(duty_pan, duty_tilt)
                    
                    
                    
                    
                
        cv2.imshow('YOLO detection results', frame)
        if cv2.waitKey(1) in [ord('q'), ord('Q')]:
            break

def main():
    # Set up the PiCamera stream with high target FPS
    stream = PiCameraStream(resolution=(640, 480), target_fps=60)
    capture_thread = threading.Thread(target=stream.update, daemon=True)
    capture_thread.start()

    # Load YOLO model; adjust the model path as needed
    model_path = "/home/XenaPi/yolo/yolo11s_ncnn_model"
    model = YOLO(model_path, task='detect')
    labels = model.names
    threshold = 0.3
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
