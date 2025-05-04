from ultralytics import YOLO
import cv2

# Load your custom-trained model
model = YOLO("bottle_detector.pt")  # Change to the path where your file is

# Open the webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Run YOLO detection on the frame
    results = model(frame)

    # Plot results
    annotated_frame = results[0].plot()

    # Show it
    cv2.imshow('YOLO Webcam Detection', annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
