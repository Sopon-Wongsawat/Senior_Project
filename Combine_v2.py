import cv2
import torch
from ultralytics import YOLO
from tracker import Tracker

# Load the first YOLOv8 model (detects only motorbikes and persons)
model1 = YOLO("yolov8n.pt")  # Change to your model (e.g., yolov8s.pt)

# Load the second YOLOv8 model for further processing
model2 = YOLO("bestV8_hemlet_v2.pt")  # Change if needed

# COCO class IDs for "person" (0) and "motorbike" (3)
TARGET_CLASSES = [0, 3]  # Check YOLOv8 COCO class list

# Open video file
#video_path = "input.mp4"
cap = cv2.VideoCapture("id4.mp4")

# Get video properties
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
fps = int(cap.get(cv2.CAP_PROP_FPS))

# Define VideoWriter to save output
output_path = "output.mp4"
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

#COLOR
colors = {"person": (0, 255, 0), "motorbike": (255, 0, 0)}

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break  # Stop if video ends

    # Run first YOLO model (detect motorbike & person)
    results1 = model1(frame)[0]
    detections = []
    
    for det in results1.boxes.data:
        x1, y1, x2, y2, conf, cls = det.tolist()
        cls = int(cls)
        
        if cls in TARGET_CLASSES:
            detections.append([x1, y1, x2, y2, conf, cls])
            
            # Draw bounding box for first model detections
            label = f"{model1.names[cls]} {conf:.2f}"
            color = colors.get(model1.names[cls], (0, 255, 255))  # Default to yellow if not found
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
            cv2.putText(frame, label, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # If at least one motorbike or person is detected, process with the second model
    if detections:
        results2 = model2(frame)[0]
        for det in results2.boxes.data:
            x1, y1, x2, y2, conf, cls = det.tolist()
            cls = int(cls)
            label = f"{model2.names[cls]} {conf:.2f}"
            color = (0, 0, 255)  # Red for second model detections
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
            cv2.putText(frame, label, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # Save frame to output video
    out.write(frame)

    # Show output
    cv2.imshow("YOLOv8 Processing", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Cleanup
cap.release()
out.release()
cv2.destroyAllWindows()

