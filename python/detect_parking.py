import cv2
from ultralytics import YOLO

parking_model = YOLO("D:/UTAR/Degree/FYP/YOLO/model/yolov8_pklot/weights/best.pt")
car_model = YOLO("yolov8n.pt")

img_path = "D:/UTAR/Degree/FYP/YOLO"
output_path = "D:/UTAR/Degree/FYP/YOLO/results/detect_result"
img = cv2.imread(img_path)

parking_results = parking_model(img_path)

# Extract parking space detections
parking_spaces = []
for detection in parking_results[0].boxes.xyxy:
    x1, y1, x2, y2 = detection[:4]
    class_id = int(detection[-1])  # 0: available, 1: occupied
    parking_spaces.append((x1, y1, x2, y2, class_id))

# Run pre-trained YOLO model to detect cars
car_results = car_model(img_path)

# Extract detected car positions
cars = []
for detection in car_results[0].boxes.xyxy:
    x1, y1, x2, y2 = detection[:4]
    label = car_results[0].names[int(detection[-1])]
    
    if label == "car":
        cars.append((x1, y1, x2, y2))

# Function to check if a car is inside a parking space
def is_inside(car, space):
    cx, cy = (car[0] + car[2]) / 2, (car[1] + car[3]) / 2  # Car center point
    return space[0] <= cx <= space[2] and space[1] <= cy <= space[3]

# Identify searching cars (cars that are NOT inside any occupied parking space)
searching_cars = []
for car in cars:
    parked = any(is_inside(car, space) and space[4] == 1 for space in parking_spaces)
    if not parked:
        searching_cars.append(car)

print(f"Detected Searching Cars: {len(searching_cars)}")

# Draw bounding boxes (green for available, red for occupied, blue for searching cars)
for x1, y1, x2, y2, class_id in parking_spaces:
    color = (0, 255, 0) if class_id == 0 else (0, 0, 255)
    cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)

for x1, y1, x2, y2 in searching_cars:
    cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)

cv2.imwrite(output_path, img)
