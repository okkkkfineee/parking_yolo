from ultralytics import YOLO

def main():
    # model = YOLO("D:/UTAR/Degree/FYP/YOLO/model/yolov8_pklot3/weights/last.pt") # Load checkpoint
    model = YOLO("yolov8s.pt")
    model.train(
        # resume=True, # Resume training
        data="D:/UTAR/Degree/FYP/YOLO/car_dataset/data.yaml",
        project="D:/UTAR/Degree/FYP/YOLO/model",
        name="yolov8_car",
        device="cuda:0",
        epochs=50,
        imgsz=640,
        batch=16,
        workers=4,
        patience=5,
        augment=True,
        hsv_h=0.015,
        hsv_s=0.5,
        hsv_v=0.4,
        degrees=10,
        translate=0.1,
        scale=0.3,
        shear=2,
        flipud=0.1,
        fliplr=0.5,
        mosaic=0.8,
        mixup=0.2,
        close_mosaic=5
    )

if __name__ == "__main__":
    main()