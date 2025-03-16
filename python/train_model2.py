from ultralytics import YOLO

def main():
    model = YOLO("D:/UTAR/Degree/FYP/YOLO/model/yolov8_pklot2/weights/last.pt") # Load checkpoint
    # model = YOLO("yolov8m.pt")
    model.train(
        resume=True, # Resume training
        data="D:/UTAR/Degree/FYP/YOLO/pklot_yolo/dataset.yaml",
        project="D:/UTAR/Degree/FYP/YOLO/model",
        name="yolov8_pklot2",
        device="cuda:0",
        epochs=50,
        imgsz=640,
        batch=16,
        workers=4,
        patience=5,
        augment=True,
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        degrees=10,
        translate=0.1,
        scale=0.5,
        shear=2,
        flipud=0.1,
        fliplr=0.5,
        mosaic=1.0,
        mixup=0.2,
        close_mosaic=5
    )

if __name__ == "__main__":
    main()