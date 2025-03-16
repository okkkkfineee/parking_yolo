from ultralytics import YOLO

def main():
    model = YOLO("D:/UTAR/Degree/FYP/YOLO/model/yolov8_pklot3/weights/best.pt") 
    model.val(
        data="D:/UTAR/Degree/FYP/YOLO/pklot_yolo/dataset.yaml",
        project="D:/UTAR/Degree/FYP/YOLO/results/val",
        name="train_model3",
        exist_ok=True,
    )

if __name__ == "__main__":
    main()