from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
import cv2
import numpy as np
import os

# Load YOLO models
model_car = YOLO("D:/UTAR/Degree/FYP/YOLO/model/yolov8_car2/weights/best.pt")
model_parking = YOLO("D:/UTAR/Degree/FYP/YOLO/model/yolov8_pklot3/weights/best.pt")

# Initialize DeepSORT tracker (for cars only)
tracker = DeepSort(max_age=70, nms_max_overlap=0.5, max_iou_distance=0.7)

# Open webcam
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)  # Set width
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)  # Set height

cv2.namedWindow("Tracking", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Tracking", 800, 450)  # Resize display window

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break  # Stop if video ends

    # Convert frame to RGB (YOLO expects RGB input)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Run YOLO models
    results_car = model_car(frame_rgb, conf=0.7, iou=0.3)  # Detect cars
    results_parking = model_parking(frame_rgb, conf=0.5, iou=0.3)  # Detect parking lots

    detections = []  # For DeepSORT tracking

    # Create overlay for better visualization
    overlay = frame.copy()

    # Process car detections (for tracking)
    for result in results_car:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])  # Ensure integer coordinates
            conf = float(box.conf[0])

            if conf < 0.7:
                continue  # Ignore low-confidence detections

            # Append in (L, T, R, B) format for DeepSORT
            detections.append(([x1, y1, x2 - x1, y2 - y1], conf, None, None))  

            # Draw RED bounding box for YOLO car detections
            cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(overlay, f'Car: {conf:.2f}', (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    # Run DeepSORT tracking
    tracks = tracker.update_tracks(detections, frame=frame.copy())  # Use unmodified frame

    # Draw tracking results
    for track in tracks:
        if track.is_confirmed() and track.time_since_update == 0:
            track_id = track.track_id  # Get track ID
            x1, y1, x2, y2 = map(int, track.to_ltrb())  # Ensure correct conversion

            # Draw tracking box (Green)
            cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # Display track ID above the box
            label = f"ID: {track_id}"
            cv2.putText(overlay, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # Process parking lot detections (not tracked)
    for result in results_parking:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])

            if conf < 0.5:
                continue

            # Draw BLUE bounding box for parking lots
            cv2.rectangle(overlay, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.putText(overlay, f'Lot: {conf:.2f}', (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    # Blend overlay for better visualization
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

    # Resize for display
    small_frame = cv2.resize(frame, (800, 450))  # Better display size
    cv2.imshow("Tracking", small_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break  # Press 'q' to quit

cap.release()
cv2.destroyAllWindows()
