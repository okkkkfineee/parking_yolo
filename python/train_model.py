from ultralytics import YOLO

def main():
    model = YOLO("yolov8n.pt")
    model.train(
        data="D:/UTAR/Degree/FYP/YOLO/pklot_yolo/dataset.yaml",
        project="D:/UTAR/Degree/FYP/YOLO/model",
        name="yolov8_pklot",
        device="cuda:0",
        epochs=100,
        imgsz=640,
        batch=16,
        workers=4,
        patience=10,
    )

if __name__ == "__main__":
    main()