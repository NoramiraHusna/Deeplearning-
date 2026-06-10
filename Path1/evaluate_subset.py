from ultralytics import YOLO

model = YOLO("best.pt")

print("Running evaluation on the 100-image subset...")
metrics = model.val(
    data="subset.yaml",
    split="val",
    plots=True
)

print("\n--- RESULTS MULTICLASS SUBSET ---")
print(f"mAP@0.5: {metrics.results_dict['metrics/mAP50(B)']:.4f}")