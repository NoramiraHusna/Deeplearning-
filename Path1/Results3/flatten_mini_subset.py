import os

# Path to your mini subset labels
label_dir = "mini_subset/labels/val"

print("Converting mini-subset labels to binary...")
for filename in os.listdir(label_dir):
    if filename.endswith(".txt"):
        filepath = os.path.join(label_dir, filename)
        
        with open(filepath, "r") as f:
            lines = f.readlines()
        
        new_lines = []
        for line in lines:
            parts = line.split()
            if parts:
                # Force the class ID to be 0 (Pothole)
                parts[0] = "0"
                new_lines.append(" ".join(parts) + "\n")
                
        with open(filepath, "w") as f:
            f.writelines(new_lines)

print("✅ Mini-subset labels successfully flattened to Class 0!")