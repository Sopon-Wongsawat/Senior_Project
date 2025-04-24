import cv2
import torch
from ultralytics import YOLO
import numpy as np
import json

# Load the first YOLOv8 model (detects only motorbikes and persons)
model1 = YOLO("yolov8n.pt")  # Change to your model (e.g., yolov8s.pt)

# Load the second YOLOv8 model for helmet detection
model2 = YOLO("bestV8_helmet_v2.pt")  # Change if needed

# COCO class IDs for "person" (0) and "motorbike" (3)
TARGET_CLASSES = [0, 3]  # Check YOLOv8 COCO class list


def load_lane_mask(json_path, frame_shape):
    with open(json_path, 'r') as f:
        data = json.load(f)

    annotated_height = data['imageHeight']
    annotated_width = data['imageWidth']
    target_height, target_width = frame_shape[:2]


    mask = np.zeros((annotated_height, annotated_width), dtype=np.uint8)

    for shape in data['shapes']:
        if shape['shape_type'] == 'polygon':
            points = np.array(shape['points'], dtype=np.int32)
            cv2.fillPoly(mask, [points], 255)

    # Resize mask to match video frame
    resized_mask = cv2.resize(mask, (target_width, target_height), interpolation=cv2.INTER_NEAREST)
    return resized_mask

# Check if a point is in the lane mask (white area)
def point_in_lane(point, mask):
    x, y = int(point[0]), int(point[1])
    if 0 <= x < mask.shape[1] and 0 <= y < mask.shape[0]:
        return mask[y, x] > 0
    return False

# Open video file
video_path = "/Users/phasin/Project_withLane/Video_1.mp4"
cap = cv2.VideoCapture(video_path)

# Get video properties
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
fps = int(cap.get(cv2.CAP_PROP_FPS))

# Define VideoWriter to save output
output_path = "output.mp4"
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))


lane_mask = load_lane_mask("/Users/phasin/Project_withLane/Lane.json", (frame_height, frame_width))

# Colors
colors = {"person": (0, 255, 0), "motorbike": (255, 0, 0)}

paused = False

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    results1 = model1(frame)[0]
    detections = []

    for det in results1.boxes.data:
        x1, y1, x2, y2, conf, cls = det.tolist()
        cls = int(cls)

        if cls in TARGET_CLASSES:
            detections.append([x1, y1, x2, y2, conf, cls])
            label_name = model1.names[cls]
            center_point = (int((x1 + x2) / 2), int(y2))  # bottom-center of box

            # Use normal color (no red for violations)
            color = colors.get(model1.names[cls], (0, 255, 255))
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
            cv2.putText(frame, label_name, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            cv2.circle(frame, center_point, 5, color, -1)

    # Run helmet detection if needed
    if detections:
        results2 = model2(frame)[0]
        for det in results2.boxes.data:
            x1, y1, x2, y2, conf, cls = det.tolist()
            cls = int(cls)
            label = f"{model2.names[cls]} {conf:.2f}"
            color = (255, 0, 130)
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
            cv2.putText(frame, label, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # Overlay lane mask (semi-transparent for visual alignment)
    # Create a colored overlay: Cyan where lane mask == 255, else black
    lane_overlay = np.zeros_like(frame)
    lane_overlay[lane_mask == 255] = (255, 0, 0)  # Cyan (B, G, R)
    frame = cv2.addWeighted(frame, 1.0, lane_overlay, 0.3, 0)

    # Save and show
    out.write(frame)
    cv2.imshow("YOLOv8 with Lane Detection", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break
    elif key == ord(" "):
        paused = not paused
        if paused:
            cv2.waitKey(0)

cap.release()
out.release()
cv2.destroyAllWindows()
