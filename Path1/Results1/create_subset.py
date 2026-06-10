import os
import random
import shutil

# --- 1. SET YOUR FOLDER PATHS HERE ---
# Point these to the folder where you extracted the zip file
source_images = r"C:\Users\nhnor\Downloads\pothole3\pothole_ground.yolov8-obb\train\images"
source_labels = r"C:\Users\nhnor\Downloads\pothole3\pothole_ground.yolov8-obb\train\labels"

# This is where the script will save the 100 selected images
dest_images = "subset_dataset/images/val"
dest_labels = "subset_dataset/labels/val"

# Make the new folders
os.makedirs(dest_images, exist_ok=True)
os.makedirs(dest_labels, exist_ok=True)

# --- 2. SETUP BALANCING LOGIC ---
# Assuming your classes in the txt files are 0 (Low), 1 (Medium), 2 (High)
class_counts = {0: 0, 1: 0, 2: 0}
target_per_class = 34 # 34 images x 3 classes = 102 total images

# Get all label files and shuffle them so the selection is random
all_labels = [f for f in os.listdir(source_labels) if f.endswith('.txt')]
random.shuffle(all_labels)

print("Scanning dataset and copying files...")

# --- 3. FILTER AND COPY ---
for label_file in all_labels:
    # Stop if we hit the target for all classes
    if all(count >= target_per_class for count in class_counts.values()):
        break

    with open(os.path.join(source_labels, label_file), 'r') as f:
        lines = f.readlines()
        
    if not lines:
        continue
        
    # Find out what classes are inside this specific image
    # YOLO format is: "class_id x y w h"
    classes_in_image = set([int(line.split()[0]) for line in lines])
    
    # Check if this image contains a class we still need more of
    needed = False
    for cls in classes_in_image:
        if cls in class_counts and class_counts[cls] < target_per_class:
            needed = True
            class_counts[cls] += 1 
            
    if needed:
        # Copy the .txt label
        shutil.copy(os.path.join(source_labels, label_file), os.path.join(dest_labels, label_file))
        
        # Copy the matching image (change .jpg to .png if your dataset uses png)
        image_file = label_file.replace('.txt', '.jpg')
        if os.path.exists(os.path.join(source_images, image_file)):
            shutil.copy(os.path.join(source_images, image_file), os.path.join(dest_images, image_file))

print("\n✅ Subset Created Successfully!")
print(f"Final count of objects per class: {class_counts}")