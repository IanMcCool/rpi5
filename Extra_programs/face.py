#!/usr/bin/env python3
# live_ncnn_detection.py
# Face detection via Ultralytics YOLO NCNN on PiCamera2 with OpenCV

import time
import cv2
import ncnn
from picamera2 import Picamera2
from ultralytics import YOLO
from adafruit_servokit import ServoKit  # added for servo control

def main():
    print("Loading YOLO NCNN model...")
    model = YOLO("/home/XenaPi/servo_projects/yolo11_face_ncnn", task="detect")
    print("Model loaded.")

    print("Configuring Picamera2...")
    picam2 = Picamera2()
    # preview configuration at 640x480
    config = picam2.create_preview_configuration(main={'size': (640, 480)})
    picam2.configure(config)
    picam2.start()
    print("Camera started.")

    # ServoKit initialization
    kit = ServoKit(channels=16)
    pan_channel = 0
    tilt_channel = 1
    pan_angle = 90  # start at center
    tilt_angle = 90
    kit.servo[pan_channel].angle = pan_angle
    kit.servo[tilt_channel].angle = tilt_angle
    Kp_pan = 0.1  # degrees per pixel
    Kp_tilt = 0.1

    window_name = 'Live NCNN Detection'
    cv2.namedWindow(window_name)

    while True:
        frame = picam2.capture_array()
        if frame is None:
            print("Warning: no frame captured, retrying...")
            time.sleep(0.1)
            continue

        # run inference
        results = model.predict(
            source=frame,
            device="cpu",
            stream=False,
            imgsz=640
        )

        # pan/tilt tracking
        if results and len(results[0].boxes.xyxy) > 0:
            # use first detected bbox
            x1, y1, x2, y2 = results[0].boxes.xyxy[0].tolist()
            cx = int((x1 + x2) / 2)
            cy = int((y1 + y2) / 2)
            error_x = cx - 320
            error_y = cy - 240
            pan_angle -= error_x * Kp_pan
            tilt_angle += error_y * Kp_tilt
            # clamp angles between 0 and 180
            pan_angle = max(0, min(180, pan_angle))
            tilt_angle = max(0, min(180, tilt_angle))
            kit.servo[pan_channel].angle = pan_angle
            kit.servo[tilt_channel].angle = tilt_angle

        # overlay detections
        for res in results:
            bboxes = res.boxes.xyxy
            confs  = res.boxes.conf

            for bbox, conf in zip(bboxes, confs):
                x1, y1, x2, y2 = map(int, bbox.tolist())
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
                cv2.putText(
                    frame,
                    f'{conf:.2f}',
                    (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0,255,0),
                    1
                )

        cv2.imshow(window_name, frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    print("Cleaning up...")
    picam2.stop()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
