from ultralytics import YOLO

model = YOLO("best.pt")

print("Running evaluation on the 30-image balanced mini subset...")
metrics = model.val(
    data="mini_subset.yaml",
    split="val",
    plots=True
)
print("✅ Done! Check your runs folder for the 3-class matrix.")