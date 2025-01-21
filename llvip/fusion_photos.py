import os
from pathlib import Path
import cv2
import numpy as np

# Define paths
base_dir = Path('./')
infrared_dir = base_dir / 'original_infrared'
visible_dir = base_dir / 'original_visible'
output_dir_stacking = base_dir / 'fusion_stacking'
output_dir_weighted_sum = base_dir / 'fusion_weighted_sum'

# Create output directories for fusion methods
for split in ['train', 'val', 'test']:
    (output_dir_stacking / split).mkdir(parents=True, exist_ok=True)
    (output_dir_weighted_sum / split).mkdir(parents=True, exist_ok=True)

# Fusion functions
def direct_stacking(rgb_image, ir_image):
    """Direct stacking (channel-wise concatenation)."""
    ir_image_resized = cv2.resize(ir_image, (rgb_image.shape[1], rgb_image.shape[0]))
    stacked_image = np.dstack((rgb_image, ir_image_resized))
    return stacked_image

def weighted_sum_fusion(rgb_image, ir_image, alpha=0.7, beta=0.3):
    """Weighted sum fusion."""
    ir_image_resized = cv2.resize(ir_image, (rgb_image.shape[1], rgb_image.shape[0]))
    ir_image_resized = cv2.normalize(ir_image_resized, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    fused_image = cv2.addWeighted(rgb_image, alpha, cv2.cvtColor(ir_image_resized, cv2.COLOR_GRAY2BGR), beta, 0)
    return fused_image

# Process each split
def process_split_for_fusion(split):
    infrared_split_dir = infrared_dir / split
    visible_split_dir = visible_dir / split
    stacking_output_split_dir = output_dir_stacking / split
    weighted_sum_output_split_dir = output_dir_weighted_sum / split

    # Process each file
    for visible_file in visible_split_dir.glob('*.jpg'):
        infrared_file = infrared_split_dir / visible_file.name
        
        if not infrared_file.exists():
            print(f"Warning: No matching infrared image found for {visible_file.name}")
            continue

        # Read images
        rgb_image = cv2.imread(str(visible_file))
        ir_image = cv2.imread(str(infrared_file), cv2.IMREAD_GRAYSCALE)

        # Apply direct stacking
        stacked_image = direct_stacking(rgb_image, ir_image)
        stacked_output_path = stacking_output_split_dir / visible_file.name
        cv2.imwrite(str(stacked_output_path), stacked_image)

        # Apply weighted sum fusion
        fused_image = weighted_sum_fusion(rgb_image, ir_image)
        weighted_sum_output_path = weighted_sum_output_split_dir / visible_file.name
        cv2.imwrite(str(weighted_sum_output_path), fused_image)

    print(f"Processed for split {split}")

# Run for train, val, and test splits
for split in ['train', 'val', 'test']:
    process_split_for_fusion(split)

print("Fusion processing completed. Check the 'fusion_stacking' and 'fusion_weighted_sum' directories.")
