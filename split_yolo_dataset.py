import os
import random
import shutil
from glob import glob

# Settings
IMAGES_DIR = "trash_labels_anotations/Images"
LABELS_DIR = "yolo_labels"
OUTPUT_DIR = "yolo_dataset"
TRAIN_RATIO = 0.8

# Create output folders
def make_dirs():
    for split in ["train", "val"]:
        os.makedirs(os.path.join(OUTPUT_DIR, "images", split), exist_ok=True)
        os.makedirs(os.path.join(OUTPUT_DIR, "labels", split), exist_ok=True)

# Get all image files and shuffle
def get_image_files():
    exts = ("*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG")
    files = []
    for ext in exts:
        files.extend(glob(os.path.join(IMAGES_DIR, ext)))
    random.shuffle(files)
    return files

# Copy images and labels to split folders
def split_and_copy():
    files = get_image_files()
    n_train = int(len(files) * TRAIN_RATIO)
    train_files = files[:n_train]
    val_files = files[n_train:]
    for split, split_files in zip(["train", "val"], [train_files, val_files]):
        for img_path in split_files:
            img_name = os.path.basename(img_path)
            label_name = os.path.splitext(img_name)[0] + ".txt"
            label_path = os.path.join(LABELS_DIR, label_name)
            out_img = os.path.join(OUTPUT_DIR, "images", split, img_name)
            out_label = os.path.join(OUTPUT_DIR, "labels", split, label_name)
            shutil.copy2(img_path, out_img)
            if os.path.exists(label_path):
                shutil.copy2(label_path, out_label)
            else:
                # If no label, create empty file (YOLO expects one)
                open(out_label, 'w').close()
    print(f"Train images: {len(train_files)}, Val images: {len(val_files)}")

if __name__ == "__main__":
    make_dirs()
    split_and_copy()
    print(f"Dataset organized in {OUTPUT_DIR}/images and {OUTPUT_DIR}/labels (train/val splits). Ready for YOLO training!")
