from ultralytics import YOLO
import cv2
import numpy as np
from sort import Sort
from collections import defaultdict

# Load YOLO models
motorcycle_model = YOLO("yolov8n.pt")  # General model for motorcycle detection
helmet_model = YOLO("bestV8_helmet_v2.pt")  # Custom model for helmet detection

# Initialize SORT tracker
mot_tracker = Sort(max_age=20, min_hits=3, iou_threshold=0.3)

# Initialize counters and trackers
motorcycle_count = 0
helmet_count = 0
tracked_objects = defaultdict(lambda: {'type': None, 'counted': False})

# Open the video file
video_path = "/Users/piyakornrodthanong/Desktop/accident videos:motor detection clips/test1vids.mp4"
cap = cv2.VideoCapture(video_path)

# Get video properties
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

# Create video writer for saving the output
output_path = "/Users/piyakornrodthanong/Project_Final/output_accident_detection.mp4"
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

def check_orientation(box):
    """Check if motorcycle is oriented upright"""
    x1, y1, x2, y2 = box
    width = x2 - x1
    height = y2 - y1
    orientation_ratio = height / width
    return orientation_ratio > 0.75

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Run motorcycle detection
    motorcycle_results = motorcycle_model(frame, conf=0.3, classes=[3])  # 3:motorcycle

    # Run helmet detection with custom model
    helmet_results = helmet_model(frame, conf=0.3)

    # Get motorcycle detections
    motorcycle_boxes = motorcycle_results[0].boxes.xyxy.cpu().numpy()
    motorcycle_classes = motorcycle_results[0].boxes.cls.cpu().numpy()
    motorcycle_conf = motorcycle_results[0].boxes.conf.cpu().numpy()

    # Get helmet detections
    helmet_boxes = helmet_results[0].boxes.xyxy.cpu().numpy()
    helmet_classes = helmet_results[0].boxes.cls.cpu().numpy()
    helmet_conf = helmet_results[0].boxes.conf.cpu().numpy()

    # Prepare detections for tracking
    detections = []
    # Add motorcycle detections
    for box, conf in zip(motorcycle_boxes, motorcycle_conf):
        detections.append([*box, conf, 3])  # 3 for motorcycle class

    # Add helmet detections
    for box, cls_id, conf in zip(helmet_boxes, helmet_classes, helmet_conf):
        detections.append([*box, conf, 44])  # 44 for helmet class

    # Update tracker
    if len(detections) > 0:
        tracked_objects_sort = mot_tracker.update(np.array(detections))
    else:
        tracked_objects_sort = np.empty((0, 6))

    # Process tracked objects
    for track in tracked_objects_sort:
        track_id = int(track[4])
        class_id = int(track[5])
        box = track[:4]
        
        # Count unique motorcycles and helmets
        if not tracked_objects[track_id]['counted']:
            if class_id == 3:  # Motorcycle
                motorcycle_count += 1
                tracked_objects[track_id] = {'type': 'motorcycle', 'counted': True}
            elif class_id == 44:  # Helmet
                helmet_count += 1
                tracked_objects[track_id] = {'type': 'helmet', 'counted': True}

        # Draw bounding boxes and labels
        color = (0, 255, 0) if class_id == 44 else (255, 0, 0)  # Green for helmet, Blue for motorcycle
        label = f"Helmet #{track_id}" if class_id == 44 else f"Motorcycle #{track_id}"
        
        # Draw detection box
        cv2.rectangle(frame, 
                     (int(box[0]), int(box[1])), 
                     (int(box[2]), int(box[3])), 
                     color, 2)
        
        # Add label with background
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        thickness = 2
        (label_width, label_height), baseline = cv2.getTextSize(label, font, font_scale, thickness)
        
        cv2.rectangle(frame, 
                     (int(box[0]), int(box[1]) - label_height - baseline - 5),
                     (int(box[0]) + label_width, int(box[1])),
                     color, -1)
        
        cv2.putText(frame, label,
                   (int(box[0]), int(box[1]) - baseline - 5),
                   font, font_scale, (255, 255, 255), thickness)

    # Add count information
    cv2.putText(frame, f"Motorcycles: {motorcycle_count}", (20, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
    cv2.putText(frame, f"Helmets: {helmet_count}", (20, 90),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Show the frame
    cv2.imshow("Object Tracking", frame)

    # Write the frame to output video
    out.write(frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release everything
cap.release()
out.release()
cv2.destroyAllWindows()