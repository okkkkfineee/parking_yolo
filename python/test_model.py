from ultralytics import YOLO

def main():
    model = YOLO("D:/UTAR/Degree/FYP/YOLO/model/yolov8_pklot3/weights/best.pt")  
    model.predict(
        source="D:/UTAR/Degree/FYP/YOLO/test2.jpg", 
        project="D:/UTAR/Degree/FYP/YOLO/results",
        name="test",
        save=True,
        exist_ok=True,
        conf=0.1,
        show=True
    )

if __name__ == "__main__":
    main()
