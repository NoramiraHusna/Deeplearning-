# test_real_world.py
from ultralytics import YOLO
import cv2
import matplotlib.pyplot as plt
import numpy as np
import os
from pathlib import Path

def test_model_on_images(model_path, image_paths, model_name):
    """Test model on multiple images and return metrics"""
    model = YOLO(model_path)
    
    results = []
    detection_counts = []
    confidence_scores = []
    
    for img_path in image_paths:
        # Run inference
        result = model(img_path, verbose=False)[0]
        
        # Count detections
        num_detections = len(result.boxes) if result.boxes else 0
        detection_counts.append(num_detections)
        
        # Get confidence scores
        if num_detections > 0:
            confs = result.boxes.conf.cpu().numpy()
            confidence_scores.extend(confs)
            avg_conf = confs.mean()
        else:
            avg_conf = 0
        
        results.append({
            'image': img_path,
            'detections': num_detections,
            'avg_confidence': avg_conf,
            'result': result
        })
    
    return {
        'name': model_name,
        'results': results,
        'total_detections': sum(detection_counts),
        'avg_detections_per_image': np.mean(detection_counts),
        'avg_confidence': np.mean(confidence_scores) if confidence_scores else 0,
        'detection_rate': sum(1 for d in detection_counts if d > 0) / len(detection_counts)
    }

def compare_models_visually(author_model, your_model, test_images):
    """Visual comparison of both models"""
    # Load models
    model1 = YOLO(author_model)
    model2 = YOLO(your_model)
    
    # Create comparison grid
    num_images = min(len(test_images), 4)
    fig, axes = plt.subplots(num_images, 2, figsize=(15, 5*num_images))
    
    for i, img_path in enumerate(test_images[:num_images]):
        # Read image
        img = cv2.imread(str(img_path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Author's model predictions
        results1 = model1(img_path, verbose=False)[0]
        axes[i, 0].imshow(img)
        axes[i, 0].set_title(f"Author's Model - {Path(img_path).name}", fontsize=12)
        axes[i, 0].axis('off')
        
        if results1.boxes:
            for box in results1.boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                conf = box.conf[0].cpu().numpy()
                rect = plt.Rectangle((x1, y1), x2-x1, y2-y1, fill=False, color='red', linewidth=2)
                axes[i, 0].add_patch(rect)
                axes[i, 0].text(x1, y1-5, f'Pothole: {conf:.2f}', color='red', fontsize=8)
        
        # Your model predictions
        results2 = model2(img_path, verbose=False)[0]
        axes[i, 1].imshow(img)
        axes[i, 1].set_title(f"Your Model - {Path(img_path).name}", fontsize=12)
        axes[i, 1].axis('off')
        
        if results2.boxes:
            for box in results2.boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                conf = box.conf[0].cpu().numpy()
                rect = plt.Rectangle((x1, y1), x2-x1, y2-y1, fill=False, color='green', linewidth=2)
                axes[i, 1].add_patch(rect)
                axes[i, 1].text(x1, y1-5, f'Pothole: {conf:.2f}', color='green', fontsize=8)
    
    plt.tight_layout()
    plt.savefig('real_world_comparison.png', dpi=150, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    # Paths to your models
    AUTHOR_MODEL = "y8best.pt"
    YOUR_MODEL = "runs\\detect\\runs\\train\\exp_epochs100\\weights\\best.pt"
    
    # Get test images - use actual pothole images
    test_images = []
    
    # Option 1: Use your video frames
    cap = cv2.VideoCapture("kaggle.mp4")
    frame_count = 0
    while len(test_images) < 10 and frame_count < 100:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_count % 30 == 0:  # Take one frame every 30 frames
            temp_path = f"temp_frame_{frame_count}.jpg"
            cv2.imwrite(temp_path, frame)
            test_images.append(temp_path)
        frame_count += 1
    cap.release()
    
    # Option 2: Or specify image files directly
    # test_images = ["pothole1.jpg", "pothole2.jpg", "pothole3.jpg"]
    
    if not test_images:
        print("No test images found!")
        exit()
    
    # Test both models
    print("Testing Author's Model...")
    author_results = test_model_on_images(AUTHOR_MODEL, test_images, "Author's Model")
    
    print("Testing Your Model...")
    your_results = test_model_on_images(YOUR_MODEL, test_images, "Your Model")
    
    # Print comparison
    print("\n" + "="*60)
    print("REAL-WORLD PERFORMANCE COMPARISON")
    print("="*60)
    print(f"{'Metric':<30s} {'Author':<15s} {'Yours':<15s}")
    print("-"*60)
    print(f"{'Detection Rate (found potholes):':<30s} {author_results['detection_rate']*100:<15.1f}% {your_results['detection_rate']*100:<15.1f}%")
    print(f"{'Avg Detections per Image:':<30s} {author_results['avg_detections_per_image']:<15.2f} {your_results['avg_detections_per_image']:<15.2f}")
    print(f"{'Avg Confidence Score:':<30s} {author_results['avg_confidence']:<15.3f} {your_results['avg_confidence']:<15.3f}")
    print(f"{'Total Detections:':<30s} {author_results['total_detections']:<15} {your_results['total_detections']:<15}")
    print("="*60)
    
    # Determine winner
    scores = {
        'Author': 0,
        'Your Model': 0
    }
    
    if your_results['detection_rate'] > author_results['detection_rate']:
        scores['Your Model'] += 1
    else:
        scores['Author'] += 1
        
    if your_results['avg_confidence'] > author_results['avg_confidence']:
        scores['Your Model'] += 1
    else:
        scores['Author'] += 1
    
    print(f"\nWinner: {max(scores, key=scores.get)}")
    
    # Visual comparison
    compare_models_visually(AUTHOR_MODEL, YOUR_MODEL, test_images)
    
    # Cleanup temp files
    for img in test_images:
        if img.startswith("temp_frame_"):
            os.remove(img)