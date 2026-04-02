from ultralytics import YOLO
import torch
import torch.serialization
import ultralytics.nn.tasks

with torch.serialization.safe_globals([ultralytics.nn.tasks.DetectionModel]):
    model = YOLO("y8best.pt")

results = model.predict(source="kaggle.mp4", show=True, device=0)
print(results)