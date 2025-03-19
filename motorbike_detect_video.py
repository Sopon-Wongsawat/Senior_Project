from ultralytics import YOLO
import cv2
import numpy as np

# Load YOLO model
model = YOLO("yolov8n.pt")

# Open the video file
video_path = "/Users/piyakornrodthanong/Desktop/accident videos:motor detection clips/test_2.mp4"
cap = cv2.VideoCapture(video_path)

# Get video properties
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

# Create video writer for saving the output
output_path = "/Users/phasin/Project_Final/output_accident_detection.mp4"
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

# Initialize previous frame
prev_frame = None
accident_detected = False
# Parameters for accident detection
ORIENTATION_THRESHOLD = 0.75  # Height/width ratio threshold for upright orientation

def check_orientation(box):
    """Check if motorcycle is oriented upright"""
    x1, y1, x2, y2 = box
    width = x2 - x1
    height = y2 - y1
    orientation_ratio = height / width
    return orientation_ratio > ORIENTATION_THRESHOLD

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Run YOLOv8 detection for motorcycles only
    results = model(frame, conf=0.3, classes=[3,0])  # 3 is motorcycle

    # Get current detections
    current_boxes = results[0].boxes.xyxy.cpu()
    current_classes = results[0].boxes.cls.cpu()
    current_conf = results[0].boxes.conf.cpu()

    # Convert frame to grayscale for motion detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Process detections
    for i, (box, cls_id, conf) in enumerate(zip(current_boxes, current_classes, current_conf)):
        box = box.tolist()
        if cls_id == 3:  # Motorcycle
            # Check motorcycle orientation
            if not check_orientation(box):
                accident_detected = True
                # Draw red box around accident
                cv2.rectangle(frame, 
                            (int(box[0]), int(box[1])), 
                            (int(box[2]), int(box[3])), 
                            (0, 0, 255), 2)
                
                # Add "Accident" label with background
                label = "Accident"
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.6
                thickness = 2
                (label_width, label_height), baseline = cv2.getTextSize(label, font, font_scale, thickness)
                
                # Draw background rectangle for text
                cv2.rectangle(frame, 
                            (int(box[0]), int(box[1]) - label_height - baseline - 5),
                            (int(box[0]) + label_width, int(box[1])),
                            (0, 0, 255), -1)
                
                # Draw text
                cv2.putText(frame, label,
                          (int(box[0]), int(box[1]) - baseline - 20),
                          font, font_scale, (255, 255, 255), thickness)

    # Update previous frame
    prev_frame = gray

    # Annotate frame with detections
    annotated_frame = results[0].plot()

    # Add accident detection status
    if accident_detected:
        cv2.putText(annotated_frame, "ACCIDENT DETECTED!", (20, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # Show the frame
    cv2.imshow("Accident Detection", annotated_frame)

    # Write the frame to output video
    out.write(annotated_frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release everything
cap.release()
out.release()
cv2.destroyAllWindows()
