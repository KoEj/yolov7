import os
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from mpl_toolkits.axes_grid1 import ImageGrid

# Define the list of 5 selected image filenames (without extensions)
selected_images = ["010265", "020065", "030408", "050115", "110190", "190576", "101039", "250201"]

# Paths to directories
rgb_dir = "./llvip/original_visible/train"
ir_dir = "./llvip/original_infrared/train"
fusion_dir = "./llvip/fusion_stacking/train"
fusion_sum_dir = "./llvip/fusion_weighted_sum/train"

# Function to load and return images
def load_images(image_list, img_dir, img_ext=".jpg"):
    images = []
    for img_name in image_list:
        img_path = os.path.join(img_dir, img_name + img_ext)
        if os.path.exists(img_path):
            images.append(np.array(Image.open(img_path)))
        else:
            print(f"Warning: {img_path} not found")
            images.append(np.zeros((224, 224, 3)))  # Placeholder if image not found
    return images

# Load images
rgb_images = load_images(selected_images, rgb_dir)
ir_images = load_images(selected_images, ir_dir)
fusion_stacking = load_images(selected_images, fusion_dir)
fusion_sum_weight = load_images(selected_images, fusion_sum_dir) 

# Create figure for ImageGrid
fig = plt.figure(figsize=(16, 6))

# ImageGrid with one extra column for labels
grid = ImageGrid(fig, 111, 
                 nrows_ncols=(4, 8),  # 3 rows (RGB, IR, Fusion) × 5 images
                 axes_pad=0.15,       # Padding between images
                 share_all=True       # Uniform axis sizes
                 )

# Add RGB images (1st row)
for ax, img in zip(grid[:8], rgb_images):
    ax.imshow(img)
    ax.axis("off")

# Add IR images (2nd row)
for ax, img in zip(grid[8:16], ir_images):
    ax.imshow(img, cmap='gray')
    ax.axis("off")

# Add Fusion images (3rd row)
for ax, img in zip(grid[16:24], fusion_stacking):
    ax.imshow(img)
    ax.axis("off")

for ax, img in zip(grid[24:], fusion_sum_weight):
    ax.imshow(img)
    ax.axis("off")

# Add centralized row labels
fig.text(0.04, 0.865, 'RGB', fontsize=16, ha='center', va='center', rotation=90)
fig.text(0.04, 0.62, 'IR', fontsize=16, ha='center', va='center', rotation=90)
fig.text(0.04, 0.38, 'DSF', fontsize=16, ha='center', va='center', rotation=90)
fig.text(0.04, 0.14, 'WSF', fontsize=16, ha='center', va='center', rotation=90)

plt.tight_layout()
plt.show()
