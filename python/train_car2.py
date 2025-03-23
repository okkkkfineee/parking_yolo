from ultralytics import YOLO

def main():
    model = YOLO("D:/UTAR/Degree/FYP/YOLO/model/yolov8_car2/weights/last.pt") # Load checkpoint
    # model = YOLO("yolov8m.pt")
    model.train(
        resume=True, # Resume training
        data="D:/UTAR/Degree/FYP/YOLO/car_dataset/data.yaml",
        project="D:/UTAR/Degree/FYP/YOLO/model",
        name="yolov8_car2",
        device="cuda:0",
        epochs=50,  # Increased epochs
        patience=10,  # Prevent early stopping
        imgsz=640,
        batch=16,
        workers=4,
        augment=True,
        lr0=0.002,  # Lower LR for stable training
        lrf=0.1,
        optimizer="AdamW",  # More adaptive optimization
        warmup_epochs=3,  # Warmup to stabilize early training
        hsv_h=0.02,
        hsv_s=0.7,
        hsv_v=0.5,
        degrees=15,
        translate=0.2,
        scale=0.4,
        shear=3,
        flipud=0.1,
        fliplr=0.5,
        mosaic=0.3,  # Lower mosaic for better generalization
        mixup=0.3,
        close_mosaic=10,  # Delay mosaic disabling
        dropout=0.2,  # Stronger dropout to prevent overfitting
    )

if __name__ == "__main__":
    main()