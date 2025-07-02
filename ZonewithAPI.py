import cv2
from ultralytics import YOLO
import numpy as np
from collections import deque, defaultdict
import gspread
from google.oauth2.service_account import Credentials
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload
import os
from datetime import datetime

# === Google Sheets and Drive Setup ===
SCOPES = [
    "https://www.googleapis.com/auth/spreadsheets",
    "https://www.googleapis.com/auth/drive"
]

try:
    creds = Credentials.from_service_account_file("credentials.json", scopes=SCOPES)
    client = gspread.authorize(creds)
    spreadsheet = client.open("Data log")
    worksheet_name = "DetectionLog"
    try:
        worksheet = spreadsheet.worksheet(worksheet_name)
    except gspread.WorksheetNotFound:
        worksheet = spreadsheet.add_worksheet(title=worksheet_name, rows="1000", cols="20")
        worksheet.append_row(["Timestamp", "Type", "Alert", "Event", "Image Link"])
    print(f"✅ Connected to Google Sheet '{spreadsheet.title}' > '{worksheet.title}'")

    # Set up Google Drive service
    drive_service = build('drive', 'v3', credentials=creds)

except Exception as e:
    print(f"❌ Error connecting to Google APIs: {e}")
    worksheet = None
    drive_service = None

# === Helper Functions ===

def save_snapshot(frame, label_type):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{label_type}_{timestamp}.jpg"
    folder = "snapshots"
    os.makedirs(folder, exist_ok=True)
    filepath = os.path.join(folder, filename)
    cv2.imwrite(filepath, frame)
    return filepath

def upload_to_drive(file_path):
    if drive_service is None:
        return None
    try:
        file_metadata = {
            'name': os.path.basename(file_path),
            'mimeType': 'image/jpeg'
        }
        media = MediaFileUpload(file_path, mimetype='image/jpeg')
        uploaded_file = drive_service.files().create(
            body=file_metadata,
            media_body=media,
            fields='id'
        ).execute()

        # Make file public
        drive_service.permissions().create(
            fileId=uploaded_file['id'],
            body={'type': 'anyone', 'role': 'reader'}
        ).execute()

        return f"https://drive.google.com/uc?id={uploaded_file['id']}"
    except Exception as e:
        print(f"❌ Drive upload failed: {e}")
        return None

def log_to_sheet(log_type, obj_id, event, image_link=None):
    if worksheet is None:
        return
    try:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        row = [timestamp, log_type, obj_id, event]
        if image_link:
            row.append(image_link)
        worksheet.append_row(row)
    except Exception as e:
        print(f"❌ Google Sheet log failed: {e}")

# === Model & Detection Setup ===
person_bike_model = YOLO("yolov8s.pt")
head_helmet_model = YOLO("new.pt")

CLASS_NAMES = {0: 'person', 3: 'motorbike'}
DISPLAY_COLORS = {
    'person': (0, 255, 0),
    'motorbike': (255, 0, 0),
    'helmet': (75, 0, 255),
    'head': (255, 190, 0)
}
CONF_THRESHOLDS = {'person': 0.3, 'motorbike': 0.3, 'helmet': 0.3, 'head': 0.3}

ZONE_LEFT_X = 300
ZONE_RIGHT_X = 460
DUPLICATE_DISTANCE_THRESHOLD = 60
recent_detections = deque(maxlen=15)

counted_ids = set()
person_bike_counts = {'person': 0, 'motorbike': 0}
helmet_count = 0
head_count = 0
track_history = defaultdict(lambda: deque(maxlen=10))

drawing = False
ix, iy = -1, -1
roi = None

# === Drawing ROI ===
def draw_roi(event, x, y, flags, param):
    global ix, iy, drawing, roi
    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        ix, iy = x, y
    elif event == cv2.EVENT_MOUSEMOVE and drawing:
        roi = (ix, iy, x, y)
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        roi = (ix, iy, x, y)

def is_recent_duplicate(x, y, label_type):
    for frame_detections in recent_detections:
        for (px, py, ptype) in frame_detections:
            if label_type == ptype and np.linalg.norm(np.array([x, y]) - np.array([px, py])) < DUPLICATE_DISTANCE_THRESHOLD:
                return True
    return False

def draw_box(frame, box, label, color):
    x1, y1, x2, y2 = map(int, box)
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

def process_head_helmet(frame):
    global helmet_count, head_count
    frame_copy = frame.copy()
    current_frame_detections = []

    cv2.line(frame_copy, (ZONE_LEFT_X, 0), (ZONE_LEFT_X, frame.shape[0]), (0, 255, 255), 2)
    cv2.line(frame_copy, (ZONE_RIGHT_X, 0), (ZONE_RIGHT_X, frame.shape[0]), (0, 255, 255), 2)

    results = head_helmet_model(frame_copy)[0]
    for det in results.boxes.data:
        x1, y1, x2, y2, conf, cls = det.tolist()
        cls = int(cls)
        label_type = 'helmet' if cls == 0 else 'head' if cls == 1 else None
        if not label_type or conf < CONF_THRESHOLDS[label_type]:
            continue

        x_center = int((x1 + x2) / 2)
        y_center = int((y1 + y2) / 2)
        if ZONE_LEFT_X <= x_center <= ZONE_RIGHT_X:
            if not is_recent_duplicate(x_center, y_center, label_type):
                if label_type == 'helmet':
                    helmet_count += 1
                else:
                    head_count += 1
                    snapshot_path = save_snapshot(frame, 'head')
                    image_link = upload_to_drive(snapshot_path)
                    log_to_sheet('Head', '🔴 Head detected', 'Counted in Zone', image_link)
                print(f"✅ Counted {label_type} at {x_center}")
            current_frame_detections.append((x_center, y_center, label_type))
        draw_box(frame_copy, (x1, y1, x2, y2), f"{label_type} {conf:.2f}", DISPLAY_COLORS[label_type])

    recent_detections.append(current_frame_detections)

    cv2.putText(frame_copy, f"Helmet count: {helmet_count}", (1000, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, DISPLAY_COLORS['helmet'], 2)
    cv2.putText(frame_copy, f"Head count: {head_count}", (1000, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, DISPLAY_COLORS['head'], 2)
    return frame_copy

# === Main Function ===
def main():
    global roi
    cap = cv2.VideoCapture("test3.mp4")
    ret, first_frame = cap.read()
    if not ret:
        print("❌ Failed to load video.")
        return

    cv2.namedWindow("Draw ROI and press ENTER")
    cv2.setMouseCallback("Draw ROI and press ENTER", draw_roi)
    clone = first_frame.copy()

    while True:
        display = clone.copy()
        if roi:
            x0, y0, x1, y1 = roi
            cv2.rectangle(display, (x0, y0), (x1, y1), (0, 255, 255), 2)
        cv2.imshow("Draw ROI and press ENTER", display)
        key = cv2.waitKey(1)
        if key == 13:  # Enter
            break
        elif key == 27:
            cap.release()
            cv2.destroyAllWindows()
            return

    cap.release()
    cv2.destroyWindow("Draw ROI and press ENTER")
    rx0, ry0, rx1, ry1 = min(roi[0], roi[2]), min(roi[1], roi[3]), max(roi[0], roi[2]), max(roi[1], roi[3])

    for result in person_bike_model.track(source="test3.mp4", tracker="botsort.yaml", conf=0.4, iou=0.3, stream=True):
        frame = result.orig_img
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

            track_history[obj_id].append((cx, cy))
            avg_cx = int(np.mean([pt[0] for pt in track_history[obj_id]]))
            avg_cy = int(np.mean([pt[1] for pt in track_history[obj_id]]))

            if len(track_history[obj_id]) >= 5:
                first_cx = track_history[obj_id][0][0]
                last_cx = track_history[obj_id][-1][0]
                moving_left = last_cx < first_cx - 10
            else:
                moving_left = False

            inside_roi = rx0 <= avg_cx <= rx1 and ry0 <= avg_cy <= ry1
            if inside_roi and moving_left and obj_id not in counted_ids:
                counted_ids.add(obj_id)
                person_bike_counts[CLASS_NAMES[cls]] += 1
                print(f"✅ Counted {CLASS_NAMES[cls]} ID {obj_id}. Total: {person_bike_counts[CLASS_NAMES[cls]]}")

            cv2.rectangle(frame, (x1b, y1b), (x2b, y2b), DISPLAY_COLORS[CLASS_NAMES[cls]], 2)
            cv2.putText(frame, f"{CLASS_NAMES[cls]} ID:{obj_id}", (x1b, y1b - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, DISPLAY_COLORS[CLASS_NAMES[cls]], 2)
            cv2.circle(frame, (avg_cx, avg_cy), 5, (0, 255, 255), -1)

        person_sum = helmet_count + head_count
        cv2.putText(frame, f"Person: {person_sum}", (1000, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, DISPLAY_COLORS['person'], 2)
        cv2.putText(frame, f"Motorbike: {person_bike_counts['motorbike']}", (1000, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, DISPLAY_COLORS['motorbike'], 2)

        frame = process_head_helmet(frame)

        cv2.imshow("Combined Detection and Tracking", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord(' '):
            print("⏸️ Paused. Press space to resume.")
            while True:
                k = cv2.waitKey(0) & 0xFF
                if k == ord(' '):
                    print("▶️ Resumed.")
                    break
                elif k == ord('q'):
                    return

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
