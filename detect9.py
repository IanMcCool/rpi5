import threading
import time
import cv2
import serial
import numpy as np
from ultralytics import YOLO
from picamera2 import Picamera2

# Serial communication setup
try:
    serial_port = serial.Serial(port='/dev/ttyACM0', baudrate=9600, timeout=1)
    print("Serial port opened successfully.")
except Exception as e:
    print("Failed to open serial port:", e)
    serial_port = None
    
# Camera and servo constants
H_FOV = 54  # degrees
V_FOV = 41  # degrees
CENTER_DEGREE = 90
FRAME_X_PIXELS = 640
FRAME_Y_PIXELS = 480
CENTER_X_PIXELS = 320
CENTER_Y_PIXELS = 240

# Servo angle limits
RIGHT_FROM_CENTER = 117  # 90 + (54/2)
LEFT_FROM_CENTER = 63    # 90 - (54/2)
TOP_FROM_CENTER = 110.5  # 90 + (41/2)
BOTTOM_FROM_CENTER = 69.5  # 90 - (41/2)

# PWM parameters
leftPulse = 0.05
centerPulse = 0.075
rightPulse = 0.10
step = 0.001
delay_ms = 20

# Pre-calculate slope values once at startup
M_PAN = (FRAME_X_PIXELS - CENTER_X_PIXELS) / (RIGHT_FROM_CENTER - CENTER_DEGREE)
M_TILT = (FRAME_Y_PIXELS - CENTER_Y_PIXELS) / (TOP_FROM_CENTER - CENTER_DEGREE)

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

def find_offset_values(xmin, ymin, xmax, ymax, frame_x, frame_y):
    """Calculate pixel offsets from center of frame"""
    box_x = (xmin + xmax) / 2
    box_y = (ymax + ymin) / 2  # Adjusted for OpenCV coordinate system
    frame_x_center = frame_x / 2
    frame_y_center = frame_y / 2
    return (frame_x_center - box_x), (frame_y_center - box_y)

def calculate_new_angles(offset_x, offset_y, current_pan, current_tilt):
    """Calculate new servo angles based on object position"""
    new_pan = current_pan - (offset_x / M_PAN)
    new_tilt = current_tilt - (offset_y / M_TILT)
    
    # Constrain to servo limits
    new_pan = max(LEFT_FROM_CENTER, min(RIGHT_FROM_CENTER, new_pan))
    new_tilt = max(BOTTOM_FROM_CENTER, min(TOP_FROM_CENTER, new_tilt))
    return new_pan, new_tilt

def angles_to_pwm(pan_angle, tilt_angle):
    """Convert angles to PWM pulse widths"""
    pan_normalized = (pan_angle - LEFT_FROM_CENTER) / (RIGHT_FROM_CENTER - LEFT_FROM_CENTER)
    tilt_normalized = (tilt_angle - BOTTOM_FROM_CENTER) / (TOP_FROM_CENTER - BOTTOM_FROM_CENTER)
    return (
        leftPulse + pan_normalized * (rightPulse - leftPulse),
        leftPulse + tilt_normalized * (rightPulse - leftPulse)
    )

def send_pwm_command(pan_pwm, tilt_pwm):
    """Send PWM values to STM32 via UART"""
    if serial_port and serial_port.is_open:
        try:
            command = f"{pan_pwm:.4f},{tilt_pwm:.4f}\n"
            serial_port.write(command.encode('utf-8'))
            return command.strip()  # Return the sent command for display
        except Exception as e:
            print("UART send error:", e)
            return None
    return None

def inference_loop(get_frame_func, model, labels, threshold, bbox_colors):
    """Main detection and tracking loop"""
    current_pan = CENTER_DEGREE
    current_tilt = CENTER_DEGREE
    last_detection_time = time.time()
    last_pwm_command = "No command sent yet"
    
    while True:
        frame = get_frame_func()
        if frame is None:
            continue

        # Reset to center if no detection for 2 seconds
        if time.time() - last_detection_time > 2.0:
            if current_pan != CENTER_DEGREE or current_tilt != CENTER_DEGREE:
                current_pan, current_tilt = CENTER_DEGREE, CENTER_DEGREE
                pan_pwm, tilt_pwm = angles_to_pwm(current_pan, current_tilt)
                cmd = send_pwm_command(pan_pwm, tilt_pwm)
                if cmd:
                    last_pwm_command = cmd

        best_detection = None
        max_confidence = threshold
        results = model(frame, verbose=False)
        
        for detection in results[0].boxes:
            xyxy = detection.xyxy.cpu().numpy().squeeze().astype(int)
            xmin, ymin, xmax, ymax = xyxy
            conf = detection.conf.item()
            classidx = int(detection.cls.item())
            classname = labels[classidx].lower()
            
            if classname == 'person' and conf > max_confidence:
                best_detection = (xmin, ymin, xmax, ymax, conf, classidx)
                max_confidence = conf

        if best_detection:
            xmin, ymin, xmax, ymax, conf, classidx = best_detection
            last_detection_time = time.time()
            
            # Calculate offsets and new angles
            offset_x, offset_y = find_offset_values(xmin, ymin, xmax, ymax, FRAME_X_PIXELS, FRAME_Y_PIXELS)
            new_pan, new_tilt = calculate_new_angles(offset_x, offset_y, current_pan, current_tilt)
            
            # Send commands only if angles changed significantly
            if abs(new_pan - current_pan) > 1 or abs(new_tilt - current_tilt) > 1:
                pan_pwm, tilt_pwm = angles_to_pwm(new_pan, new_tilt)
                cmd = send_pwm_command(pan_pwm, tilt_pwm)
                if cmd:
                    last_pwm_command = cmd
                current_pan, current_tilt = new_pan, new_tilt

            # Draw detection info
            color = bbox_colors[classidx % len(bbox_colors)]
            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), color, 2)
            label = f'Person: {int(conf * 100)}%'
            labelSize = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
            cv2.rectangle(frame, (xmin, ymin - labelSize[1] - 10),
                        (xmin + labelSize[0], ymin), color, cv2.FILLED)
            cv2.putText(frame, label, (xmin, ymin - 7),
                      cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
            
            # Draw crosshair at frame center
            cv2.drawMarker(frame, (CENTER_X_PIXELS, CENTER_Y_PIXELS), 
                         (0, 255, 0), cv2.MARKER_CROSS, 20, 2)
        
        # Display PWM values in the top-left corner
        cv2.putText(frame, f"Last PWM Command: {last_pwm_command}", (10, 30),
                  cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        
        # Display current angles
        cv2.putText(frame, f"Pan: {current_pan:.1f}° Tilt: {current_tilt:.1f}°", (10, 60),
                  cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        cv2.imshow('Person Tracking with PWM Output', frame)
        if cv2.waitKey(1) in [ord('q'), ord('Q')]:
            break

def main():
    # Set up the PiCamera stream
    stream = PiCameraStream(resolution=(FRAME_X_PIXELS, FRAME_Y_PIXELS), target_fps=60)
    capture_thread = threading.Thread(target=stream.update, daemon=True)
    capture_thread.start()

    # Load YOLO model
    model_path = "/home/XenaPi/yolo/yolov8n_ncnn_model"
    model = YOLO(model_path, task='detect')
    labels = model.names
    threshold = 0.5  # Increased confidence threshold
    bbox_colors = [
        (164, 120, 87), (68, 148, 228), (93, 97, 209), (178, 182, 133), (88, 159, 106),
        (96, 202, 231), (159, 124, 168), (169, 162, 241), (98, 118, 150), (172, 176, 184)
    ]

    # Start inference thread
    inference_thread = threading.Thread(
        target=inference_loop,
        args=(stream.read, model, labels, threshold, bbox_colors),
        daemon=True
    )
    inference_thread.start()

    # Main loop
    try:
        while inference_thread.is_alive():
            time.sleep(0.1)
    except KeyboardInterrupt:
        pass
    finally:
        stream.stop()
        cv2.destroyAllWindows()
        if serial_port and serial_port.is_open:
            serial_port.close()
        print("Program terminated cleanly")

if __name__ == "__main__":
    main()
