import os
import random
import shutil

# --- SET YOUR PATHS ---
# Point this to where your dataset is saved
source_images = r"C:\Users\nhnor\Downloads\pothole3\subset_dataset\images\val" 
source_labels = r"C:\Users\nhnor\Downloads\pothole3\subset_dataset\labels\val"

dest_images = "mini_subset/images/val"
dest_labels = "mini_subset/labels/val"

os.makedirs(dest_images, exist_ok=True)
os.makedirs(dest_labels, exist_ok=True)

class_counts = {0: 0, 1: 0, 2: 0}
target_per_class = 10 # 10 data for each class

all_labels = [f for f in os.listdir(source_labels) if f.endswith('.txt')]
random.shuffle(all_labels)

print("Scanning for 30 balanced images...")

for label_file in all_labels:
    if all(count >= target_per_class for count in class_counts.values()):
        break

    with open(os.path.join(source_labels, label_file), 'r') as f:
        lines = f.readlines()
        
    if not lines:
        continue
        
    classes_in_image = set([int(line.split()[0]) for line in lines])
    
    needed = False
    for cls in classes_in_image:
        if cls in class_counts and class_counts[cls] < target_per_class:
            needed = True
            class_counts[cls] += 1 
            
    if needed:
        shutil.copy(os.path.join(source_labels, label_file), os.path.join(dest_labels, label_file))
        image_file = label_file.replace('.txt', '.jpg')
        if os.path.exists(os.path.join(source_images, image_file)):
            shutil.copy(os.path.join(source_images, image_file), os.path.join(dest_images, image_file))

print(f"✅ Mini Subset Created! Final count: {class_counts}")