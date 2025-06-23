import cv2
from ultralytics import YOLO

# --- CONFIG ---
ZONE_LEFT_X = 250
ZONE_RIGHT_X = 450

# Class mappings for each model
PERSON_MOTORBIKE_CLASSES = {0: 'person', 3: 'motorbike'}
HELMET_HEAD_CLASSES = {0: 'helmet', 1: 'head'}
COLORS = {
    "person": (0, 255, 0),
    "motorbike": (255, 0, 0),
    "helmet": (75, 0, 255),
    "head": (255, 190, 0)
}

# --- ROI selection for person/motorbike ---
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

def main():
    global ix, iy, drawing, roi

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
    x0, y0, x1, y1 = roi
    rx0, ry0, rx1, ry1 = min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1)
    last_centers = {}
    counted_ids = {'person': set(), 'motorbike': set()}
    counts = {'person': 0, 'motorbike': 0, 'helmet': 0, 'head': 0}

    # --- Load models ---
    person_model = YOLO("yolov8s.pt")
    helmet_model = YOLO("new.pt")

    cap = cv2.VideoCapture("Test1vids.mp4")
    cv2.namedWindow("Detection")

    # --- For helmet/head tracking ---
    helmet_head_last_centers = {}
    helmet_head_counted_ids = {'helmet': set(), 'head': set()}

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # --- Person & Motorbike detection and counting (BoT-SORT) ---
        results = person_model.track(frame, tracker="botsort.yaml", conf=0.3, iou=0.5)
        if len(results) > 0:
            result = results[0]
            cv2.rectangle(frame, (rx0, ry0), (rx1, ry1), (0, 255, 255), 2)
            for box in result.boxes:
                if box.id is None:
                    continue
                cls = int(box.cls.cpu().item())
                obj_id = int(box.id.cpu().item())
                if cls not in PERSON_MOTORBIKE_CLASSES:
                    continue

                x1b, y1b, x2b, y2b = map(int, box.xyxy.cpu().numpy()[0])
                cx = (x1b + x2b) // 2
                cy = (y1b + y2b) // 2

                prev = last_centers.get(obj_id)
                moving_right = prev is not None and cx > prev[0]
                if moving_right:
                    last_centers[obj_id] = (cx, cy)
                    continue

                # Draw box and ID
                cv2.rectangle(frame, (x1b, y1b), (x2b, y2b), COLORS[PERSON_MOTORBIKE_CLASSES[cls]], 2)
                cv2.putText(frame, f"{PERSON_MOTORBIKE_CLASSES[cls]} ID:{obj_id}", (x1b, y1b - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLORS[PERSON_MOTORBIKE_CLASSES[cls]], 2)
                cv2.circle(frame, (cx, cy), 5, (0, 255, 255), -1)

                inside_roi = rx0 <= cx <= rx1 and ry0 <= cy <= ry1
                moving_left = prev is not None and cx < prev[0]

                if inside_roi and moving_left and obj_id not in counted_ids[PERSON_MOTORBIKE_CLASSES[cls]]:
                    counted_ids[PERSON_MOTORBIKE_CLASSES[cls]].add(obj_id)
                    counts[PERSON_MOTORBIKE_CLASSES[cls]] += 1
                    print(f"✅ Counted {PERSON_MOTORBIKE_CLASSES[cls]} ID {obj_id}. Total {PERSON_MOTORBIKE_CLASSES[cls]}: {counts[PERSON_MOTORBIKE_CLASSES[cls]]}")

                last_centers[obj_id] = (cx, cy)

        # --- Helmet & Head detection and tracking (BoT-SORT) ---
        helmet_results = helmet_model.track(frame, tracker="botsort.yaml", conf=0.3, iou=0.5)
        if len(helmet_results) > 0:
            result = helmet_results[0]
            # Draw vertical zone lines
            cv2.line(frame, (ZONE_LEFT_X, 0), (ZONE_LEFT_X, frame.shape[0]), (0, 255, 255), 2)
            cv2.line(frame, (ZONE_RIGHT_X, 0), (ZONE_RIGHT_X, frame.shape[0]), (0, 255, 255), 2)
            cv2.putText(frame, "COUNTING ZONE", (ZONE_LEFT_X + 5, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

            for box in result.boxes:
                if box.id is None:
                    continue
                cls = int(box.cls.cpu().item())
                obj_id = int(box.id.cpu().item())
                if cls not in HELMET_HEAD_CLASSES:
                    continue

                x1b, y1b, x2b, y2b = map(int, box.xyxy.cpu().numpy()[0])
                cx = (x1b + x2b) // 2
                cy = (y1b + y2b) // 2

                prev = helmet_head_last_centers.get(obj_id)
                moving_right = prev is not None and cx > prev[0]
                if moving_right:
                    helmet_head_last_centers[obj_id] = (cx, cy)
                    continue

                moving_left = prev is not None and cx < prev[0]
                in_zone = ZONE_LEFT_X <= cx <= ZONE_RIGHT_X

                # Draw box and ID
                cv2.rectangle(frame, (x1b, y1b), (x2b, y2b), COLORS[HELMET_HEAD_CLASSES[cls]], 2)
                cv2.putText(frame, f"{HELMET_HEAD_CLASSES[cls]} ID:{obj_id}", (x1b, y1b - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLORS[HELMET_HEAD_CLASSES[cls]], 2)
                cv2.circle(frame, (cx, cy), 5, (0, 255, 255), -1)

                if in_zone and moving_left and obj_id not in helmet_head_counted_ids[HELMET_HEAD_CLASSES[cls]]:
                    helmet_head_counted_ids[HELMET_HEAD_CLASSES[cls]].add(obj_id)
                    counts[HELMET_HEAD_CLASSES[cls]] += 1
                    print(f"✅ Counted {HELMET_HEAD_CLASSES[cls]} ID {obj_id}. Total {HELMET_HEAD_CLASSES[cls]}: {counts[HELMET_HEAD_CLASSES[cls]]}")

                helmet_head_last_centers[obj_id] = (cx, cy)

        # --- Display counts ---
        H, W = frame.shape[:2]
        y_offset = 30
        for i, cls in enumerate(['person', 'motorbike', 'helmet', 'head']):
            cv2.putText(frame, f"{cls.title()}: {counts[cls]}", (W - 220, y_offset + i * 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLORS.get(cls, (255,255,255)), 2)

        cv2.imshow("Detection", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord(' '):
            print("Paused. Press space to resume.")
            while True:
                k = cv2.waitKey(0) & 0xFF
                if k == ord(' '):
                    print("Resumed.")
                    break
                elif k == ord('q'):
                    exit()

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()