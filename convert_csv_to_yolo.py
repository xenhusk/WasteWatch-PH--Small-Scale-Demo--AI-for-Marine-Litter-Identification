import os
import pandas as pd
from PIL import Image

# Paths
ANNOTATIONS_CSV = "trash_labels_anotations/annotations.csv"
IMAGES_DIR = "trash_labels_anotations/Images"
YOLO_LABELS_DIR = "yolo_labels"

# 1. Read CSV
ann = pd.read_csv(ANNOTATIONS_CSV)

# 2. Get class list and assign class ids
df_classes = ann['class_name'].unique().tolist()
class_to_id = {cls: i for i, cls in enumerate(df_classes)}

# 3. Create output dir
os.makedirs(YOLO_LABELS_DIR, exist_ok=True)

# 4. Group by image and write YOLO txt files
for img_name, group in ann.groupby('image_name'):
    img_path = os.path.join(IMAGES_DIR, img_name)
    try:
        with Image.open(img_path) as im:
            w, h = im.size
    except Exception as e:
        print(f"Could not open {img_path}: {e}")
        continue
    yolo_lines = []
    for _, row in group.iterrows():
        class_id = class_to_id[row['class_name']]
        # Convert to YOLO format (normalized)
        x_min, y_min, x_max, y_max = row[['x_min','y_min','x_max','y_max']]
        x_center = ((x_min + x_max) / 2) / w
        y_center = ((y_min + y_max) / 2) / h
        bbox_w = (x_max - x_min) / w
        bbox_h = (y_max - y_min) / h
        yolo_lines.append(f"{class_id} {x_center:.6f} {y_center:.6f} {bbox_w:.6f} {bbox_h:.6f}")
    # Write label file
    label_path = os.path.join(YOLO_LABELS_DIR, os.path.splitext(img_name)[0] + ".txt")
    with open(label_path, 'w') as f:
        f.write("\n".join(yolo_lines))

# 5. Write classes.txt
with open(os.path.join(YOLO_LABELS_DIR, "classes.txt"), 'w') as f:
    for cls in df_classes:
        f.write(cls + "\n")

print(f"Done! YOLO label files are in {YOLO_LABELS_DIR}/. Class list: {df_classes}")
