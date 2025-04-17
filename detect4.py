import threading
import time
import cv2
import serial
import numpy as np
from ultralytics import YOLO
from picamera2 import Picamera2

# Initialize UART with timeout to prevent blocking
uart = serial.Serial('/dev/ttyAMA0', baudrate=9600, timeout=0.1)

# Global variable to store feedback coordinates
feedback_coords = None
feedback_lock = threading.Lock()

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
    offset_y = frame_y_center - box_y
    
    #return and cast as int for pwm values
    return int(offset_x), int(offset_y)

# Function to read feedback from the Nucleo microcontroller
def read_feedback():
    global feedback_coords
    
    while True:
        try:
            if uart.in_waiting > 0:
                line = uart.readline().decode('utf-8').strip()
                
                # Check if the line contains the feedback message
                if "Coordinates received:" in line:
                    # Extract coordinates from the message
                    coords_str = line.replace("Coordinates received:", "").strip()
                    
                    try:
                        # Parse the coordinates
                        x, y = map(float, coords_str.split(','))
                        with feedback_lock:
                            feedback_coords = (x, y)
                    except (ValueError, Exception) as e:
                        print(f"Error parsing coordinates: {e}")
        except Exception as e:
            print(f"Error reading from UART: {e}")
            
        time.sleep(0.01)  # Small delay to prevent CPU hogging
        
def inference_loop(get_frame_func, model, labels, threshold, bbox_colors):
    global feedback_coords
    
    while True:
        frame = get_frame_func()
        if frame is None:
            continue
            
        # Run inference
        results = model(frame, verbose=False)
        detections = results[0].boxes
        
        # Get frame dimensions
        frame_height, frame_width = frame.shape[:2]
        
        # Draw received feedback coordinates on the frame
        with feedback_lock:
            if feedback_coords is not None:
                fb_x, fb_y = feedback_coords
                # Convert from offsets to actual screen coordinates
                screen_x = int(frame_width/2 - fb_x)
                screen_y = int(frame_height/2 - fb_y)
                
                # Draw a crosshair at the feedback coordinates
                cv2.drawMarker(frame, (screen_x, screen_y), (0, 0, 255), 
                               markerType=cv2.MARKER_CROSS, markerSize=20, thickness=2)
                
                # Add text label for feedback
                cv2.putText(frame, f"Feedback: ({int(fb_x)}, {int(fb_y)})", 
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        # Draw detections
        for detection in detections:
            xyxy_tensor = detection.xyxy.cpu()
            xyxy = xyxy_tensor.numpy().squeeze()  # [xmin, ymin, xmax, ymax]
            xmin, ymin, xmax, ymax = xyxy.astype(int)
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
                    offset_x, offset_y = center_bound(xmin, ymin, xmax, ymax, frame_width, frame_height)
                    
                    center_x = (xmin + xmax) / 2
                    center_y = (ymin + ymax) / 2
                    
                    cv2.circle(frame, (int(center_x), int(center_y)), 5, (0, 255, 0), -1)
                    print(f"Person detected. Offset (x, y): ({offset_x}, {offset_y})")
                    
                    # Add text label for detected person
                    cv2.putText(frame, f"Detected: ({offset_x}, {offset_y})", 
                                (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    
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

    # Start the feedback reading thread
    feedback_thread = threading.Thread(target=read_feedback, daemon=True)
    feedback_thread.start()

    # Load YOLO model; adjust the model path as needed
    model_path = "/home/XenaPi/yolo/yolov8n_ncnn_model"
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
