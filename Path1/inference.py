from ultralytics import YOLO
import cv2
import numpy as np
import time

# 1. Load your custom trained model
model = YOLO("best.pt") 

# 2. Point to your specific video file 
video_path = r"C:\Users\nhnor\Downloads\pothole3\mixkit-potholes-in-a-rural-road-25208-hd-ready.mp4"

# 3. Open the video using OpenCV
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Error: Could not open the video file. Check the path!")
    exit()

# Get the height, width, and FPS of the video frame
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps_video = int(cap.get(cv2.CAP_PROP_FPS))
horizon_y = frame_height // 2  # Assume the horizon is roughly in the middle

# --- SETUP: VIDEO SAVING ---
# This will save the final result to a file named 'final_pothole_result.mp4' in your folder
fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
out = cv2.VideoWriter('final_pothole_result.mp4', fourcc, fps_video, (frame_width, frame_height))

# --- SETUP: ANALYTICS TRACKERS ---
start_time = time.time()
total_frames = 0
total_detections = 0

all_hazard_scores = []
all_depth_scores = []
all_size_scores = []
severity_counts = {"MINOR": 0, "MODERATE": 0, "SEVERE": 0}

print("Starting Smart Pothole System... Press 'q' on the video window to stop early.")

# 4. Main Video Loop
while cap.isOpened():
    success, frame = cap.read()
    
    if success:
        total_frames += 1
        
        # Run YOLO with 30% confidence to prevent overlapping "ghost" boxes
        results = model(frame, stream=True, verbose=False, conf=0.30)
        
        for result in results:
            for box in result.boxes:
                # Extract pixel coordinates
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                
                # --- FEATURE 1: PERSPECTIVE-AWARE SCALING ---
                raw_width = x2 - x1
                raw_height = y2 - y1
                raw_area = raw_width * raw_height
                
                # Prevent math errors if the box is above the horizon
                safe_y2 = max(y2, horizon_y + 1) 
                
                # Calculate multiplier: further away = bigger multiplier
                perspective_multiplier = (frame_height / safe_y2) ** 2
                true_size_score = raw_area * perspective_multiplier
                
                # Normalize size score to a 0-100 scale
                size_factor = min(true_size_score / 50000.0, 1.0) * 100

                # --- FEATURE 2: SHADOW PROFILING (PSEUDO-DEPTH) ---
                shadow_factor = 0
                crop = frame[y1:y2, x1:x2] 
                
                if crop.size > 0: 
                    gray_crop = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
                    mean, stddev = cv2.meanStdDev(gray_crop)
                    contrast = stddev[0][0] 
                    
                    # Normalize depth score to 0-100 
                    shadow_factor = min(contrast / 40.0, 1.0) * 100

                # --- CUSTOM SEVERITY MATRIX ---
                hazard_score = (size_factor * 0.6) + (shadow_factor * 0.4)

                if hazard_score > 70:
                    severity = "SEVERE"
                    color = (0, 0, 255) # Red
                elif hazard_score > 40:
                    severity = "MODERATE"
                    color = (0, 165, 255) # Orange
                else:
                    severity = "MINOR"
                    color = (0, 255, 0) # Green
                
                # --- VISUAL FEEDBACK ---
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                label = f"{severity} (Hazard: {int(hazard_score)})"
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

                # --- RECORD ANALYTICS ---
                total_detections += 1
                severity_counts[severity] += 1
                all_hazard_scores.append(hazard_score)
                all_depth_scores.append(shadow_factor)
                all_size_scores.append(size_factor)

        # Write the processed frame to our save file
        out.write(frame)
        
        # Display the frame live on screen
        cv2.imshow("Smart Pothole Detection", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        break

# 5. Clean Up Process
cap.release()
out.release() # Safely close the saved video file
cv2.destroyAllWindows()

# 6. Generate the Terminal Data Report
end_time = time.time()
total_time = end_time - start_time
fps = total_frames / total_time if total_time > 0 else 0

print("\n" + "="*50)
print(" 🏁 POTHOLE DETECTION SYSTEM: RUN SUMMARY 🏁 ")
print("="*50)

print(f"⏱️  Performance Metrics:")
print(f"   - Total Processing Time: {total_time:.2f} seconds")
print(f"   - Total Frames Processed: {total_frames}")
print(f"   - Average Speed: {fps:.2f} FPS (Frames Per Second)")

if total_detections > 0:
    print(f"\n📊 Detection Analytics (Total Box Detections: {total_detections}):")
    print(f"   - MINOR Hazards:    {severity_counts['MINOR']}")
    print(f"   - MODERATE Hazards: {severity_counts['MODERATE']}")
    print(f"   - SEVERE Hazards:   {severity_counts['SEVERE']}")
    
    print(f"\n📏 System Logic Extremes & Averages:")
    print(f"   - Max Hazard Score: {max(all_hazard_scores):.0f}/100 (Avg: {sum(all_hazard_scores)/len(all_hazard_scores):.0f})")
    print(f"   - Max Depth Score:  {max(all_depth_scores):.0f}/100 (Avg: {sum(all_depth_scores)/len(all_depth_scores):.0f})")
    print(f"   - Max Size Score:   {max(all_size_scores):.0f}/100 (Avg: {sum(all_size_scores)/len(all_size_scores):.0f})")
else:
    print("\n⚠️ No potholes were detected in this video.")

print("="*50 + "\n")
print("✅ Saved final video to 'final_pothole_result.mp4' in your current folder.")