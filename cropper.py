import cv2
import numpy as np
import os
from pathlib import Path

def center_crop(image, crop_width, crop_height):
    height, width = image.shape[:2]
    
    # Calculate crop coordinates
    start_x = (width - crop_width) // 2
    start_y = (height - crop_height) // 2
    
    # Crop the center of the image
    cropped = image[start_y:start_y+crop_height, start_x:start_x+crop_width]
    
    return cropped, (start_x, start_y)

def process_directory(input_dir, output_dir, crop_width, crop_height):
    # Create output directory if it doesn't exist
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Process all images in directory
    start_coords = None
    for filename in os.listdir(input_dir):
        if filename.endswith(('.PNG', '.JPG', '.jpeg')):
            # Read image
            img_path = os.path.join(input_dir, filename)
            img = cv2.imread(img_path)
            
            if img is None:
                print(f"Failed to load {filename}")
                continue
                
            # Crop image
            cropped_img, coords = center_crop(img, crop_width, crop_height)
            
            # Save starting coordinates for the first image
            if start_coords is None:
                start_coords = coords
            
            # Save cropped image
            output_path = os.path.join(output_dir, filename)
            cv2.imwrite(output_path, cropped_img)
            
    return start_coords

if __name__ == "__main__":
    # Configuration
    input_directory = "images_unused"
    output_directory = "croppedColor"
    crop_width = 4460  # Desired width
    crop_height = 8736  # Desired height
    
    # Process images
    start_x, start_y = process_directory(input_directory, output_directory, 
                                       crop_width, crop_height)
    
    print(f"Images cropped successfully")
    print(f"Crop started at x={start_x}, y={start_y}")
    print(f"Use these values to adjust your K matrix in C++")