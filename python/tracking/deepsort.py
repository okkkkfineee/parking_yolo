from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
import cv2

# Load YOLO model
model = YOLO("../yolov8m.pt")

# Initialize DeepSORT tracker
tracker = DeepSort(max_age=30)

cap = cv2.VideoCapture("your_video.mp4")  # Load video
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break  # Stop if video ends

    # Run YOLO detection
    results = model(frame)

    detections = []
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            cls = int(box.cls[0])
            detections.append(([x1, y1, x2, y2], conf, cls))

    # Run tracking
    tracks = tracker.update_tracks(detections, frame=frame)

    # Draw tracking results
    for track in tracks:
        if not track.is_confirmed():
            continue  # Skip unconfirmed tracks

        track_id = track.track_id
        ltrb = track.to_ltrb()
        x1, y1, x2, y2 = map(int, ltrb)

        # Draw bounding box and track ID
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f'ID: {track_id}', (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Show the frame
    cv2.imshow("Tracking", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break  # Press 'q' to quit

cap.release()
cv2.destroyAllWindows()
