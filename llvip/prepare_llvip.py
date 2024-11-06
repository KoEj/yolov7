import os
import random
import shutil
from pathlib import Path
import xml.etree.ElementTree as ET

# Paths definition
base_dir = Path('./')
annotations_dir = base_dir / 'Annotations'
visible_dir = base_dir / 'visible'
infrared_dir = base_dir / 'infrared'
yolo_labels_dir = base_dir / 'labels' 
output_txt_dir = base_dir

for split in ['train', 'val', 'test']:
    (yolo_labels_dir / split).mkdir(parents=True, exist_ok=True)
    (visible_dir / split).mkdir(parents=True, exist_ok=True)
    (infrared_dir / split).mkdir(parents=True, exist_ok=True)

train_ratio = 0.7
val_ratio = 0.15
test_ratio = 0.15

all_visible_images = sorted(list(visible_dir.rglob('*.jpg')))
if not all_visible_images:
    print(f"No images found in {visible_dir}. Please check the directory path.")
else:
    print(f"Found {len(all_visible_images)} images in {visible_dir}.")

# Step 1: Split images based on visible images
def split_images(image_files, train_ratio, val_ratio, test_ratio):
    random.shuffle(image_files)
    train_count = int(len(image_files) * train_ratio)
    val_count = int(len(image_files) * val_ratio)

    train_files = image_files[:train_count]
    val_files = image_files[train_count:train_count + val_count]
    test_files = image_files[train_count + val_count:]

    return train_files, val_files, test_files

visible_train, visible_val, visible_test = split_images(all_visible_images, train_ratio, val_ratio, test_ratio)

# Step 2: Move visible and infrared images based on matching file names and generate labels
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

def find_infrared_file(filename):
    for split in ['train', 'val', 'test']:
        infrared_path = infrared_dir / split / filename
        if infrared_path.exists():
            return infrared_path
    return None

def process_split(visible_files, split_name):
    for file in visible_files:
        visible_dest = visible_dir / split_name / file.name
        infrared_source = find_infrared_file(file.name)
        split_infrared_dest = infrared_dir / split_name / file.name
        label_dest = yolo_labels_dir / split_name / f"{file.stem}.txt"

        if not visible_dest.exists():
            shutil.move(file, visible_dest)

        if infrared_source and not split_infrared_dest.exists():
            shutil.move(infrared_source, split_infrared_dest)
        elif not infrared_source:
            print(f"Warning: Matching infrared image for {file.name} not found in any infrared split folder.")

        annotation_file = annotations_dir / f"{file.stem}.xml"
        if annotation_file.exists():
            convert_voc_to_yolo(annotation_file, label_dest, classes)

process_split(visible_train, 'train')
process_split(visible_val, 'val')
process_split(visible_test, 'test')

# Step 3: Create train.txt, val.txt, and test.txt with relative paths
def create_image_list_file(image_files, output_file):
    with open(output_file, 'w') as f:
        for img in image_files:
            relative_path = os.path.relpath(img, './')
            f.write(f"./{relative_path}\n")
    print(f"{output_file} created.")

# Collect images for each split and write to list files
train_visible_files = list((visible_dir / 'train').rglob('*.jpg'))
val_visible_files = list((visible_dir / 'val').rglob('*.jpg'))
test_visible_files = list((visible_dir / 'test').rglob('*.jpg'))

create_image_list_file(train_visible_files, output_txt_dir / 'visible_train.txt')
create_image_list_file(val_visible_files, output_txt_dir / 'visible_val.txt')
create_image_list_file(test_visible_files, output_txt_dir / 'visible_test.txt')

# Same for infrared images
train_infrared_files = list((infrared_dir / 'train').rglob('*.jpg'))
val_infrared_files = list((infrared_dir / 'val').rglob('*.jpg'))
test_infrared_files = list((infrared_dir / 'test').rglob('*.jpg'))

create_image_list_file(train_infrared_files, output_txt_dir / 'infrared_train.txt')
create_image_list_file(val_infrared_files, output_txt_dir / 'infrared_val.txt')
create_image_list_file(test_infrared_files, output_txt_dir / 'infrared_test.txt')

print("Dataset splitting, annotation conversion, and list file creation completed.")
