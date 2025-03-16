import json
import os

# Define paths
input_base_path = "pklot_coco_json"
output_base_path = "pklot_yolo"
datasets = ["train", "test", "valid"]  # Folders to process

# Process each dataset (train, test, valid)
for dataset in datasets:
    input_json_path = os.path.join(input_base_path, dataset, "_annotations.coco.json")
    output_dir = os.path.join(output_base_path, dataset)

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Load COCO JSON
    with open(input_json_path, "r") as f:
        data = json.load(f)

    category_mapping = {cat["id"]: idx for idx, cat in enumerate(data["categories"])}

    # Process each image
    for image in data["images"]:
        image_id = image["id"]
        image_w, image_h = image["width"], image["height"]
        image_filename = os.path.splitext(image["file_name"])[0]
        yolo_txt_path = os.path.join(output_dir, f"{image_filename}.txt")

        # Find annotations for this image
        annotations = [ann for ann in data["annotations"] if ann["image_id"] == image_id]

        with open(yolo_txt_path, "w") as f:
            for ann in annotations:
                category_id = category_mapping[ann["category_id"]]
                x, y, w, h = ann["bbox"]

                x_center = (x + w / 2) / image_w
                y_center = (y + h / 2) / image_h
                w /= image_w
                h /= image_h

                # Write annotation
                f.write(f"{category_id} {x_center:.6f} {y_center:.6f} {w:.6f} {h:.6f}\n")

    print(f"Converted {dataset} set successfully! YOLO annotations saved in '{output_dir}'.")

print("COCO to YOLO conversion completed for all datasets.")
