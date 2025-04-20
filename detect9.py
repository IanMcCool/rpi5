import threading
import time
import cv2
import serial
import numpy as np
from ultralytics import YOLO
from picamera2 import Picamera2

# ── SERIAL COMMUNICATION SETUP ─────────────────────────────────────────────
try:
    serial_port = serial.Serial(port='/dev/ttyACM0', baudrate=9600, timeout=1)
    print("Serial port opened successfully.")
except Exception as e:
    print("Failed to open serial port:", e)
    serial_port = None

# ── CAMERA AND SERVO CONSTANTS ────────────────────────────────────────────
H_FOV = 54  # degrees
V_FOV = 41  # degrees
CENTER_DEGREE = 90
FRAME_X_PIXELS = 640
FRAME_Y_PIXELS = 480
CENTER_X_PIXELS = FRAME_X_PIXELS // 2
CENTER_Y_PIXELS = FRAME_Y_PIXELS // 2

# ── SERVO ANGLE LIMITS ───────────────────────────────────────────────────
RIGHT_FROM_CENTER = 117  # 90 + (54/2)
LEFT_FROM_CENTER = 63    # 90 - (54/2)
TOP_FROM_CENTER = 110.5  # 90 + (41/2)
BOTTOM_FROM_CENTER = 69.5  # 90 - (41/2)

# ── PWM PARAMETERS ──────────────────────────────────────────────────────
leftPulse = 0.05
centerPulse = 0.075
rightPulse = 0.10
step = 0.001
delay_ms = 20

# ── TRACKING PARAMETERS ────────────────────────────────────────────────
# Pre-calculate slope values once at startup for efficiency
M_PAN = (FRAME_X_PIXELS - CENTER_X_PIXELS) / (RIGHT_FROM_CENTER - CENTER_DEGREE)
M_TILT = (FRAME_Y_PIXELS - CENTER_Y_PIXELS) / (TOP_FROM_CENTER - CENTER_DEGREE)

# Smoothing factor (0 = no smoothing, 1 = no movement)
SMOOTHING_FACTOR = 0.3
SIGNIFICANT_ANGLE_CHANGE = 0.5  # Minimum angle change to send a command

# Detection parameters
MIN_DETECTION_CONFIDENCE = 0.5
LOST_OBJECT_TIMEOUT = 2.0  # Seconds before resetting to center

class PiCameraStream:
    """Thread-safe camera stream handler for Raspberry Pi Camera"""
    def __init__(self, resolution=(FRAME_X_PIXELS, FRAME_Y_PIXELS), target_fps=60):
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
        """Thread function to continuously update the frame buffer"""
        while not self.stopped:
            frame_bgra = self.camera.capture_array()
            frame = cv2.cvtColor(frame_bgra, cv2.COLOR_BGRA2BGR)
            with self.lock:
                self.frame = frame

    def read(self):
        """Thread-safe frame retrieval"""
        with self.lock:
            return self.frame.copy() if self.frame is not None else None

    def stop(self):
        """Clean shutdown of camera"""
        self.stopped = True
        self.camera.stop()

def find_offset_values(xmin, ymin, xmax, ymax):
    """
    Calculate pixel offsets from center of frame to object center
    Returns: (offset_x, offset_y) where positive values mean the object
    needs to move right/up to be centered
    """
    box_x = (xmin + xmax) / 2
    box_y = (ymin + ymax) / 2
    
    # Calculate offsets (positive when object is off-center)
    offset_x = CENTER_X_PIXELS - box_x  # Positive when object is to the right
    offset_y = CENTER_Y_PIXELS - box_y  # Positive when object is below center
    
    return offset_x, offset_y

def calculate_new_angles(offset_x, offset_y, current_pan, current_tilt):
    """
    Calculate new servo angles based on object position using PID approach
    with smoothing to prevent jerky motion
    """
    # Calculate raw angle adjustments
    # Negate offset_x because positive offset (object to right) means we need to increase angle (turn right)
    # Negate offset_y because positive offset (object below center) means we need to increase angle (tilt down)
    raw_pan = current_pan - (offset_x / M_PAN)
    raw_tilt = current_tilt - (offset_y / M_TILT)
    
    # Apply smoothing (weighted average between current and calculated angles)
    smooth_pan = current_pan * SMOOTHING_FACTOR + raw_pan * (1 - SMOOTHING_FACTOR)
    smooth_tilt = current_tilt * SMOOTHING_FACTOR + raw_tilt * (1 - SMOOTHING_FACTOR)
    
    # Constrain to servo limits
    new_pan = max(LEFT_FROM_CENTER, min(RIGHT_FROM_CENTER, smooth_pan))
    new_tilt = max(BOTTOM_FROM_CENTER, min(TOP_FROM_CENTER, smooth_tilt))
    
    return new_pan, new_tilt

def angles_to_pwm(pan_angle, tilt_angle):
    """Convert angles to PWM pulse widths"""
    # Map pan angle from [LEFT_FROM_CENTER, RIGHT_FROM_CENTER] to [leftPulse, rightPulse]
    pan_normalized = (pan_angle - LEFT_FROM_CENTER) / (RIGHT_FROM_CENTER - LEFT_FROM_CENTER)
    # Map tilt angle from [BOTTOM_FROM_CENTER, TOP_FROM_CENTER] to [leftPulse, rightPulse]
    tilt_normalized = (tilt_angle - BOTTOM_FROM_CENTER) / (TOP_FROM_CENTER - BOTTOM_FROM_CENTER)
    
    # Calculate PWM values
    pan_pwm = leftPulse + pan_normalized * (rightPulse - leftPulse)
    tilt_pwm = leftPulse + tilt_normalized * (rightPulse - leftPulse)
    
    return pan_pwm, tilt_pwm

def send_servo_command(pan_angle, tilt_angle):
    """Convert angles to PWM and send to STM32 via UART"""
    if serial_port and serial_port.is_open:
        try:
            # Convert to PWM values
            pan_pwm, tilt_pwm = angles_to_pwm(pan_angle, tilt_angle)
            
            # Format command as expected by STM32
            command = f"{pan_pwm:.4f},{tilt_pwm:.4f}\n"
            serial_port.write(command.encode('utf-8'))
            return True
        except Exception as e:
            print(f"Error sending command: {e}")
    return False

def draw_tracking_info(frame, xmin, ymin, xmax, ymax, offset_x, offset_y, 
                      current_pan, current_tilt, confidence, color):
    """Draw bbox, tracking info, and debug visualizations on frame"""
    # Draw bounding box
    cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), color, 2)
    
    # Draw label with confidence
    label = f'Person: {int(confidence * 100)}%'
    label_size, baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    y_label = max(ymin, label_size[1] + 10)
    cv2.rectangle(frame, (xmin, y_label - label_size[1] - 10),
                (xmin + label_size[0], y_label + baseline - 10), color, cv2.FILLED)
    cv2.putText(frame, label, (xmin, y_label - 7),
              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    
    # Draw centers and crosshair
    object_center = (int((xmin + xmax) / 2), int((ymin + ymax) / 2))
    
    # Draw frame center crosshair
    cv2.drawMarker(frame, (CENTER_X_PIXELS, CENTER_Y_PIXELS), 
                 (0, 255, 0), cv2.MARKER_CROSS, 20, 2)
    
    # Draw object center point
    cv2.circle(frame, object_center, 5, (0, 0, 255), -1)
    
    # Draw line from center to object
    cv2.line(frame, (CENTER_X_PIXELS, CENTER_Y_PIXELS), object_center, (255, 0, 0), 2)
    
    # Show offset and angle info
    cv2.putText(frame, f"Offset X: {int(offset_x)}px", (10, 30), 
              cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.putText(frame, f"Offset Y: {int(offset_y)}px", (10, 60), 
              cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.putText(frame, f"Pan: {current_pan:.2f}°", (10, 90), 
              cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.putText(frame, f"Tilt: {current_tilt:.2f}°", (10, 120), 
              cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

def inference_loop(get_frame_func, model, labels, bbox_colors):
    """Main detection and tracking loop"""
    current_pan = CENTER_DEGREE
    current_tilt = CENTER_DEGREE
    last_detection_time = time.time()
    frame_count = 0
    start_time = time.time()
    skip_frames = 0  # Process every nth frame for slower devices
    
    while True:
        frame = get_frame_func()
        if frame is None:
            continue
        
        frame_count += 1
        elapsed = time.time() - start_time
        if elapsed >= 1.0:
            fps = frame_count / elapsed
            # Dynamically adjust frame skipping based on performance
            if fps < 15 and skip_frames < 2:
                skip_frames += 1
            elif fps > 25 and skip_frames > 0:
                skip_frames -= 1
            frame_count = 0
            start_time = time.time()
        
        # Skip frames if needed (for slower devices)
        if skip_frames > 0 and frame_count % (skip_frames + 1) != 0:
            cv2.putText(frame, f"FPS: {fps:.1f} (skipping frames)", (10, FRAME_Y_PIXELS - 10),
                      cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
            cv2.imshow('Person Tracking', frame)
            if cv2.waitKey(1) in [ord('q'), ord('Q')]:
                break
            continue
        
        # Reset to center if no detection for timeout period
        if time.time() - last_detection_time > LOST_OBJECT_TIMEOUT:
            if abs(current_pan - CENTER_DEGREE) > 2 or abs(current_tilt - CENTER_DEGREE) > 2:
                print("No detection - resetting to center")
                current_pan = CENTER_DEGREE
                current_tilt = CENTER_DEGREE
                send_servo_command(current_pan, current_tilt)

        # Find best person detection
        best_detection = None
        max_confidence = MIN_DETECTION_CONFIDENCE
        
        # Run inference
        results = model(frame, verbose=False)
        
        # Process detections
        for detection in results[0].boxes:
            xyxy = detection.xyxy.cpu().numpy().squeeze().astype(int)
            xmin, ymin, xmax, ymax = xyxy
            conf = detection.conf.item()
            classidx = int(detection.cls.item())
            classname = labels[classidx].lower()
            
            # Only track people with confidence above threshold
            if classname == 'person' and conf > max_confidence:
                best_detection = (xmin, ymin, xmax, ymax, conf, classidx)
                max_confidence = conf

        # If we found a good detection
        if best_detection:
            xmin, ymin, xmax, ymax, conf, classidx = best_detection
            last_detection_time = time.time()
            
            # Calculate offsets
            offset_x, offset_y = find_offset_values(xmin, ymin, xmax, ymax)
            
            # Calculate new angles with smoothing
            new_pan, new_tilt = calculate_new_angles(offset_x, offset_y, current_pan, current_tilt)
            
            # Only send commands if angles changed significantly (reduces servo jitter)
            if (abs(new_pan - current_pan) > SIGNIFICANT_ANGLE_CHANGE or 
                abs(new_tilt - current_tilt) > SIGNIFICANT_ANGLE_CHANGE):
                if send_servo_command(new_pan, new_tilt):
                    current_pan, current_tilt = new_pan, new_tilt

            # Draw all tracking info
            color = bbox_colors[classidx % len(bbox_colors)]
            draw_tracking_info(frame, xmin, ymin, xmax, ymax, offset_x, offset_y,
                              current_pan, current_tilt, conf, color)
        else:
            # No detection - show status
            cv2.putText(frame, "No person detected", (10, 30), 
                      cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        # Add FPS display
        if elapsed >= 1.0:
            cv2.putText(frame, f"FPS: {fps:.1f}", (10, FRAME_Y_PIXELS - 10),
                      cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
        
        # Show the frame
        cv2.imshow('Person Tracking', frame)
        if cv2.waitKey(1) in [ord('q'), ord('Q')]:
            break

def main():
    """Main program entry point"""
    print("Starting person tracking system...")
    
    # Set up the PiCamera stream
    stream = PiCameraStream(resolution=(FRAME_X_PIXELS, FRAME_Y_PIXELS), target_fps=60)
    capture_thread = threading.Thread(target=stream.update, daemon=True)
    capture_thread.start()
    print("Camera initialized")

    # Load YOLO model
    model_path = "/home/XenaPi/yolo/yolov8n_ncnn_model"
    model = YOLO(model_path, task='detect')
    labels = model.names
    bbox_colors = [
        (164, 120, 87), (68, 148, 228), (93, 97, 209), (178, 182, 133), (88, 159, 106),
        (96, 202, 231), (159, 124, 168), (169, 162, 241), (98, 118, 150), (172, 176, 184)
    ]
    print("Model loaded successfully")

    # Start inference thread
    inference_thread = threading.Thread(
        target=inference_loop,
        args=(stream.read, model, labels, bbox_colors),
        daemon=True
    )
    inference_thread.start()
    print("Tracking started - press 'q' to quit")

    # Main loop - keep alive until user quits
    try:
        while inference_thread.is_alive():
            time.sleep(0.1)
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    finally:
        # Clean shutdown
        stream.stop()
        cv2.destroyAllWindows()
        if serial_port and serial_port.is_open:
            # Reset servos to center position before exit
            print("Resetting servos to center")
            send_servo_command(CENTER_DEGREE, CENTER_DEGREE)
            time.sleep(0.5)  # Give servos time to move
            serial_port.close()
        print("Program terminated cleanly")

if __name__ == "__main__":
    main()
