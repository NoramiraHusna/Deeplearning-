from ultralytics import YOLO

# Load the open-source binary model
model = YOLO("best2.pt")

print("Running binary evaluation on balanced 30-image subset...")
# Run validation with plotting enabled to generate curves and matrices
metrics = model.val(
    data="mini_binary.yaml",
    split="val",
    plots=True
)

print("\n--- EVALUATION RESULTS ---")
print(f"Precision (P):  {metrics.results_dict['metrics/precision(B)']:.4f}")
print(f"Recall (R):     {metrics.results_dict['metrics/recall(B)']:.4f}")
print(f"mAP@0.5:        {metrics.results_dict['metrics/mAP50(B)']:.4f}")

# F1-score can be read directly from the generated BoxF1_curve.png
print("\n✅ Evaluation complete! Check the 'runs/detect/val' folder for your confusion matrix and F1 curve.")