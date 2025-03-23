import cv2
import os
from ultralytics import YOLO

# Load YOLO models
model_path = os.path.abspath("D:/UTAR/Degree/FYP/YOLO/model/yolov8_car2/weights/best.pt")
model_car = YOLO(model_path)

# Open webcam
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)  # Set width
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)  # Set height

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break  # Stop if video ends

    # Convert frame to RGB (YOLO expects RGB input)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Run car detection model with lower resolution for speed
    results_car = model_car(frame_rgb, conf=0.6, iou=0.3)

    # Create overlay for better visualization
    overlay = frame.copy()

    # Process car detections
    for result in results_car:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            cls = int(box.cls[0])  # Class index (0 for Car)

            if conf < 0.6:  # Confidence threshold
                continue

            # Draw bounding box for cars (RED)
            cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(overlay, f'Car: {conf:.2f}', (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    # Blend overlay for better visualization
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

    # Resize for display
    small_frame = cv2.resize(frame, (800, 450))  # Better display size
    cv2.imshow("YOLOv8 Car Detection", small_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break  # Press 'q' to quit

cap.release()
cv2.destroyAllWindows()
