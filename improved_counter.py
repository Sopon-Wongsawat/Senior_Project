import cv2
from ultralytics import YOLO

# Initialize count and tracking
counted_ids = set()
counts = {'person': 0, 'motorbike': 0}
track_history = {}         # obj_id: list of (cx, cy)
disqualified_ids = set()   # obj_id that ever moved righttrack_history = {}         # obj_id: list of (cx, cy)
disqualified_ids = set()   # obj_id that ever moved right
# Class mappings for YOLOv8 default COCO classes
CLASS_NAMES = {0: 'person', 3: 'motorbike'}
COLORS = {0: (0, 255, 0), 3: (255, 0, 0)}

# ROI drawing variables
drawing = False
ix, iy = -1, -1
roi = None

def draw_roi(event, x, y, flags, param):
    global ix, iy, drawing, roi
    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        ix, iy = x, y
        roi = None
    elif event == cv2.EVENT_MOUSEMOVE and drawing:
        roi = (ix, iy, x, y)
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        roi = (ix, iy, x, y)

# Load YOLOv8 model

model = YOLO("yolov8s.pt")

# --- ROI selection phase ---
cap = cv2.VideoCapture("Test1vids.mp4")
ret, first_frame = cap.read()
if not ret:
    print("Failed to read video.")
    exit()

clone = first_frame.copy()
cv2.namedWindow("Draw ROI and press ENTER")
cv2.setMouseCallback("Draw ROI and press ENTER", draw_roi)

while True:
    display = clone.copy()
    if roi is not None:
        x0, y0, x1, y1 = roi
        cv2.rectangle(display, (x0, y0), (x1, y1), (0, 255, 255), 2)
    cv2.imshow("Draw ROI and press ENTER", display)
    key = cv2.waitKey(1)
    if key == 13:  # ENTER key
        break
    elif key == 27:  # ESC to quit
        cap.release()
        cv2.destroyAllWindows()
        exit()

cv2.destroyWindow("Draw ROI and press ENTER")
cap.release()

# --- Detection phase ---
cv2.namedWindow("BoT-SORT Zone Counter")
# roi is already set from above
x0, y0, x1, y1 = roi
rx0, ry0, rx1, ry1 = min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1)
last_centers = {}
for result in model.track(source="Test1vids.mp4", tracker="botsort.yaml", conf=0.3, iou=0.5, stream=True):
    frame = result.orig_img

    # Draw ROI
    cv2.rectangle(frame, (rx0, ry0), (rx1, ry1), (0, 255, 255), 2)

    for box in result.boxes:
        if box.id is None:
            continue
        cls = int(box.cls.cpu().item())
        obj_id = int(box.id.cpu().item())
        if cls not in CLASS_NAMES:
            continue

        x1b, y1b, x2b, y2b = map(int, box.xyxy.cpu().numpy()[0])
        cx = (x1b + x2b) // 2
        cy = (y1b + y2b) // 2

        # Draw box and ID
        cv2.rectangle(frame, (x1b, y1b), (x2b, y2b), COLORS[cls], 2)
        cv2.putText(frame, f"{CLASS_NAMES[cls]} ID:{obj_id}", (x1b, y1b - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLORS[cls], 2)
        cv2.circle(frame, (cx, cy), 5, (0, 255, 255), -1)

        # Only count if:
        # - previous center was outside ROI on the right
        # - current center is inside ROI
        # - not already counted
        prev = last_centers.get(obj_id)
        inside_roi = rx0 <= cx <= rx1 and ry0 <= cy <= ry1
        moving_left = prev is not None and cx < prev[0]

        if inside_roi and moving_left and obj_id not in counted_ids:
            counted_ids.add(obj_id)
            counts[CLASS_NAMES[cls]] += 1
            print(f"✅ Counted {CLASS_NAMES[cls]} ID {obj_id}. Total {CLASS_NAMES[cls]}: {counts[CLASS_NAMES[cls]]}")

        # Update last center for this ID
        last_centers[obj_id] = (cx, cy)

        
    # Display counts on the right
    H, W = frame.shape[:2]
    y_offset = 30
    for i, cls in enumerate(['person', 'motorbike']):
        cv2.putText(frame, f"{cls.title()}: {counts[cls]}", (W - 220, y_offset + i * 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    cv2.imshow("BoT-SORT Zone Counter", frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord(' '):  # Space to pause
        print("Paused. Press space to resume.")
        while True:
            k = cv2.waitKey(0) & 0xFF
            if k == ord(' '):  # Space to resume
                print("Resumed.")
                break
            elif k == ord('q'):
                exit()

cv2.destroyAllWindows()