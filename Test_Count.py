import cv2
from ultralytics import YOLO
import numpy as np
from collections import deque

# Load YOLO models
yolo_model = YOLO("yolov8n.pt")  
head_and_helmet_model = YOLO("best17_6_2-25.pt")  

# Targets
TARGET_CLASSES = [0, 3]  # person=0, motorcycle=3

CONF_THRESHOLDS = {
    'person': 0.15,
    'motorbike': 0.15,
    'helmet': 0.15,
    'head': 0.15
}

COLORS = {
    "person": (0, 255, 0),
    "motorbike": (255, 0, 0),
    "helmet": (75, 0, 255),
    "head": (255, 190, 0)
}

LINE_X = 150
LINE_X2 = 600  # Vertical line position for counting
helmet_count = 0
head_count = 0

recent_detections = deque(maxlen=10)  # each item is a list of (x, y, type)


def draw_box(frame, box, label, color):
    x1, y1, x2, y2 = map(int, box)
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)


def is_recent_duplicate(x, y, label_type):
    for frame_detections in recent_detections:
        for (px, py, ptype) in frame_detections:
            if label_type == ptype and np.linalg.norm(np.array([x, y]) - np.array([px, py])) < 20:
                return True
    return False


def process_frame(frame):
    global helmet_count, head_count

    # Draw yellow vertical line
    cv2.line(frame, (LINE_X, 0), (LINE_X, frame.shape[0]), (0, 255, 255), 2)
    #cv2.line(frame, (LINE_X2, 0), (LINE_X2, frame.shape[0]), (0, 255, 255), 2)

    # Store detections for this frame
    current_frame_detections = []

    ### Detect person & motorcycle ###
    yolo_results = yolo_model(frame, classes=TARGET_CLASSES)[0]
    for det in yolo_results.boxes.data:
        x1, y1, x2, y2, conf, cls = det.tolist()
        cls = int(cls)
        class_type = 'person' if cls == 0 else 'motorbike'
        if conf >= CONF_THRESHOLDS[class_type]:
            label = f"{class_type} {conf:.2f}"
            draw_box(frame, (x1, y1, x2, y2), label, COLORS[class_type])

    ### Detect helmet & head ###
    head_results = head_and_helmet_model(frame)[0]
    for det in head_results.boxes.data:
        x1, y1, x2, y2, conf, cls = det.tolist()
        cls = int(cls)

        if cls == 0:
            label_type = 'helmet'
        elif cls == 1:
            label_type = 'head'
        else:
            continue

        if conf >= CONF_THRESHOLDS[label_type]:
            label = f"{label_type.capitalize()} {conf:.2f}"
            color = COLORS[label_type]
            draw_box(frame, (x1, y1, x2, y2), label, color)

            x_center = int((x1 + x2) / 2)
            y_center = int((y1 + y2) / 2)

            # If passed the line and not seen in last 3 frames → count
            if x_center < LINE_X:
                if not is_recent_duplicate(x_center, y_center, label_type):
                    if label_type == 'helmet':
                        helmet_count += 1
                    else:
                        head_count += 1
                    print(f"✅ Counted new {label_type} at ({x_center},{y_center})")

            # Save to current frame memory
            current_frame_detections.append((x_center, y_center, label_type))

    # Add this frame's detections to recent memory
    recent_detections.append(current_frame_detections)

    # Show counts
    cv2.putText(frame, f"Helmet count: {helmet_count}", (1000, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, COLORS['helmet'], 2)
    cv2.putText(frame, f"Head count: {head_count}", (1000, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, COLORS['head'], 2)

    return frame


def main():
    video_path = "/Users/phasin/Project_withLane/test3.mp4"
    cap = cv2.VideoCapture(video_path)

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    out = cv2.VideoWriter('output_test.mp4',
                          cv2.VideoWriter_fourcc(*'mp4v'),
                          fps, (width, height))

    paused = False

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        processed_frame = process_frame(frame)
        out.write(processed_frame)
        cv2.imshow('Helmet & Head Detection', processed_frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord(' '):
            paused = not paused
            if paused:
                cv2.waitKey(0)

    cap.release()
    out.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
