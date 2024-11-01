import os
import random
import shutil
from pathlib import Path
import xml.etree.ElementTree as ET

# Paths definition
base_dir = Path('./')  # Update to your base directory
annotations_dir = base_dir / 'Annotations'
visible_dir = base_dir / 'visible'
infrared_dir = base_dir / 'infrared'
yolo_labels_dir = base_dir / 'labels'  # Directory to store YOLO format labels
output_txt_dir = base_dir  # Output directory for train.txt, val.txt, test.txt

# Ensure YOLO labels directory exists
yolo_labels_dir.mkdir(parents=True, exist_ok=True)

# Create directories for train, val, and test splits for both visible and infrared images
for split in ['train', 'val', 'test']:
    for image_dir in [visible_dir / split, infrared_dir / split]:
        if image_dir.exists():
            shutil.rmtree(image_dir)  # Clear previous split directories
        image_dir.mkdir(parents=True, exist_ok=True)

# Parameters for the split
train_ratio = 0.7
val_ratio = 0.15
test_ratio = 0.15

# Step 1: Split images into train, val, and test without overlap
def split_images(image_dir, train_ratio, val_ratio, test_ratio):
    image_files = sorted(list(image_dir.rglob('*.jpg')))
    if not image_files:
        print(f"No images found in {image_dir}. Please check the directory path.")
        return [], [], []

    random.shuffle(image_files)
    train_count = int(len(image_files) * train_ratio)
    val_count = int(len(image_files) * val_ratio)

    train_files = image_files[:train_count]
    val_files = image_files[train_count:train_count + val_count]
    test_files = image_files[train_count + val_count:]

    return train_files, val_files, test_files

# Split visible and infrared images independently
visible_train, visible_val, visible_test = split_images(visible_dir, train_ratio, val_ratio, test_ratio)
infrared_train, infrared_val, infrared_test = split_images(infrared_dir, train_ratio, val_ratio, test_ratio)

# Ensure directories for train, val, and test exist without deleting contents
for split in ['train', 'val', 'test']:
    (visible_dir / split).mkdir(parents=True, exist_ok=True)
    (infrared_dir / split).mkdir(parents=True, exist_ok=True)

#  Move files to respective directories
def move_files(files, split_dir):
    for file in files:
        dest = split_dir / file.name
        if not dest.exists():
            shutil.move(file, dest)

# Move files for visible and infrared sets
move_files(visible_train, visible_dir / 'train')
move_files(visible_val, visible_dir / 'val')
move_files(visible_test, visible_dir / 'test')

move_files(infrared_train, infrared_dir / 'train')
move_files(infrared_val, infrared_dir / 'val')
move_files(infrared_test, infrared_dir / 'test')


# Step 2: Convert VOC annotations to YOLO format
classes = ['person']

def convert_voc_to_yolo(voc_file, yolo_file, classes):
    tree = ET.parse(voc_file)
    root = tree.getroot()
    img_width = int(root.find('size/width').text)
    img_height = int(root.find('size/height').text)

    with open(yolo_file, 'w') as f:
        for obj in root.findall('object'):
            cls_name = obj.find('name').text
            if cls_name not in classes:
                continue  # Skip undefined classes

            cls_id = classes.index(cls_name)
            xml_box = obj.find('bndbox')
            x_min = int(xml_box.find('xmin').text)
            y_min = int(xml_box.find('ymin').text)
            x_max = int(xml_box.find('xmax').text)
            y_max = int(xml_box.find('ymax').text)

            # Convert to YOLO format
            x_center = (x_min + x_max) / 2 / img_width
            y_center = (y_min + y_max) / 2 / img_height
            width = (x_max - x_min) / img_width
            height = (y_max - y_min) / img_height

            f.write(f"{cls_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")

for xml_file in annotations_dir.glob('*.xml'):
    yolo_label_path = yolo_labels_dir / f"{xml_file.stem}.txt"
    convert_voc_to_yolo(xml_file, yolo_label_path, classes)


# Step 3: Create train.txt, val.txt, and test.txt with relative paths
def create_image_list_file(image_files, output_file):
    with open(output_file, 'w') as f:
        for img in image_files:
            relative_path = os.path.relpath(img, './')
            f.write(f"./{relative_path}\n")
    print(f"{output_file} created.")

# Visible
train_visible_files = list((visible_dir / 'train').glob('*.jpg'))
val_visible_files = list((visible_dir / 'val').glob('*.jpg'))
test_visible_files = list((visible_dir / 'test').glob('*.jpg'))

create_image_list_file(train_visible_files, output_txt_dir / 'visible_train.txt')
create_image_list_file(val_visible_files, output_txt_dir / 'visible_val.txt')
create_image_list_file(test_visible_files, output_txt_dir / 'visible_test.txt')

# Infrared
train_infrared_files = list((infrared_dir / 'train').glob('*.jpg'))
val_infrared_files = list((infrared_dir / 'val').glob('*.jpg'))
test_infrared_files = list((infrared_dir / 'test').glob('*.jpg'))

create_image_list_file(train_infrared_files, output_txt_dir / 'infrared_train.txt')
create_image_list_file(val_infrared_files, output_txt_dir / 'infrared_val.txt')
create_image_list_file(test_infrared_files, output_txt_dir / 'infrared_test.txt')

print("Dataset splitting, annotation conversion, and list file creation completed.")
