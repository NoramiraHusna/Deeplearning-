from ultralytics import YOLO
import cv2
import matplotlib.pyplot as plt
import matplotlib.animation as animation

model = YOLO("C:\\Users\\ahmad\\Desktop\\fypee\\pothole\\best_seg.pt")
cap   = cv2.VideoCapture("C:\\Users\\ahmad\\Desktop\\fypee\\pothole\\kaggle.mp4", cv2.CAP_FFMPEG)

fig, ax = plt.subplots(figsize=(10, 8))
plt.title("Pothole Segmentation")
plt.axis("off")
im = ax.imshow([[0]])  # placeholder

def update(frame):
    ret, frame = cap.read()
    if not ret:
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # loop video
        return
    
    results  = model(frame, task="segment", conf=0.5, verbose=False)[0]
    annotated = results.plot()
    annotated = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
    im.set_data(annotated)
    im.set_extent([0, annotated.shape[1], annotated.shape[0], 0])
    return [im]

ani = animation.FuncAnimation(fig, update, interval=30, blit=True)
plt.tight_layout()
plt.show()

cap.release()