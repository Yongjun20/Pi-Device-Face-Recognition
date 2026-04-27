from picamera2 import Picamera2
import face_recognition
import pickle
import cv2
import time
import os
import numpy as np
from datetime import datetime, timedelta
import json
import shutil
from threading import Thread

ENCODINGS_FILE = "encodings.pickle"
CASCADE_PATH = "/usr/share/opencv4/haarcascades/haarcascade_frontalface_default.xml"

# ----------------------------
# Server path (fixed mount point)
# ----------------------------
SERVER_PATH = "/mnt/pcshare"

USER_JSON_FILE = "user.json"
LOG_FOLDER = "LOG"
LOGIN_INTERVAL = 1  # minutes
last_sent_times = {}

# ----------------------------
# Load / save local JSON
# ----------------------------
def load_users():
    if not os.path.exists(USER_JSON_FILE):
        return {}
    with open(USER_JSON_FILE, "r") as f:
        return json.load(f)


def save_users(data):
    with open(USER_JSON_FILE, "w") as f:
        json.dump(data, f, indent=4)

# ----------------------------
# Sync user.json bi-directionally
# ----------------------------
def sync_user_json():
    try:
        local_file = USER_JSON_FILE
        server_file = os.path.join(SERVER_PATH, USER_JSON_FILE)

        if not os.path.exists(SERVER_PATH):
            return

        if os.path.exists(server_file):
            local_time = os.path.getmtime(local_file)
            server_time = os.path.getmtime(server_file)

            if server_time > local_time:
                shutil.copy2(server_file, local_file)
                print("[SYNC] Pulled user.json from PC")

            elif local_time > server_time:
                shutil.copy2(local_file, server_file)
                print("[SYNC] Pushed user.json to PC")

        else:
            shutil.copy2(local_file, server_file)
            print("[SYNC] Created user.json on PC")

    except Exception as e:
        print(f"[SYNC ERROR] {e}")

# ----------------------------
# Write log file (sync to PC)
# ----------------------------
def write_log(message):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    date_str = datetime.now().strftime("%Y-%m-%d")

    os.makedirs(LOG_FOLDER, exist_ok=True)
    local_log = os.path.join(LOG_FOLDER, f"{date_str}.log")

    line = f"[{timestamp}] {message}\n"

    # local log
    with open(local_log, "a") as f:
        f.write(line)

    # server log
    server_log = os.path.join(SERVER_PATH, f"{date_str}.log")
    if os.path.exists(SERVER_PATH) and os.access(SERVER_PATH, os.W_OK):
        try:
            with open(server_log, "a") as f:
                f.write(line)
        except Exception as e:
            print(f"[WARN] Server log failed: {e}")

# ----------------------------
# Handle recognition event
# ----------------------------
def handle_recognition(name, confidence):
    if name.lower() == "unknown":
        return

    now = datetime.now()

    if name in last_sent_times:
        if now - last_sent_times[name] < timedelta(minutes=LOGIN_INTERVAL):
            return

    last_sent_times[name] = now

    users = load_users()

    if name not in users:
        users[name] = {
            "first_registered": now.strftime("%Y-%m-%d %H:%M:%S"),
            "last_login": now.strftime("%Y-%m-%d %H:%M:%S")
        }
    else:
        users[name]["last_login"] = now.strftime("%Y-%m-%d %H:%M:%S")

    save_users(users)

    # 🔥 Sync across devices
    sync_user_json()

    log_msg = f"Recognized: {name} | Confidence: {round(confidence * 100, 1)}%"
    write_log(log_msg)
    print(f"[INFO] {log_msg}")

# ----------------------------
# Frame reader (keeps latest frame)
# ----------------------------
class FrameReader:
    def __init__(self, picam2):
        self.picam2 = picam2
        self.frame = None
        self.running = True
        self._thread = Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self):
        while self.running:
            try:
                self.frame = self.picam2.capture_array()
            except:
                pass

    def read(self):
        return self.frame

    def stop(self):
        self.running = False
        self._thread.join(timeout=2)

# ----------------------------
# Main recognition
# ----------------------------
def start_recognition(frame_callback=None, stop_flag=None):

    currentname = "unknown"

    if not os.path.exists(ENCODINGS_FILE):
        print("[ERROR] encodings.pickle not found.")
        return

    print("[INFO] Loading encodings + detector...")

    if os.path.exists(SERVER_PATH):
        print("[INFO] Server connected.")
    else:
        print("[WARN] Server not mounted.")

    # sync at startup
    sync_user_json()

    with open(ENCODINGS_FILE, "rb") as f:
        data = pickle.load(f)

    detector = cv2.CascadeClassifier(CASCADE_PATH)

    picam2 = Picamera2()
    picam2.configure(
        picam2.create_preview_configuration(
            main={"format": "XRGB8888", "size": (640, 480)}
        )
    )
    picam2.start()
    time.sleep(1.5)

    reader = FrameReader(picam2)

    try:
        frame_count = 0

        while True:
            if stop_flag and stop_flag():
                break

            frame = reader.read()
            if frame is None:
                continue

            frame_count += 1

            if frame_count % 3 != 0:
                if frame_callback:
                    frame_callback(frame, [])
                continue

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            rects = detector.detectMultiScale(gray, 1.2, 5, minSize=(30, 30))
            boxes = [(y, x + w, y + h, x) for (x, y, w, h) in rects]

            encodings = face_recognition.face_encodings(rgb, boxes)
            names = []

            for encoding in encodings:
                distances = face_recognition.face_distance(data["encodings"], encoding)
                best_match_index = np.argmin(distances)
                confidence = max(0, 1 - distances[best_match_index])

                matches = face_recognition.compare_faces(
                    data["encodings"], encoding, tolerance=0.27
                )

                if True in matches and confidence >= 0.70:
                    name = data["names"][best_match_index]
                else:
                    name = "Unknown"

                if currentname != name:
                    currentname = name
                    handle_recognition(currentname, confidence)

                names.append((name, confidence))

            for ((top, right, bottom, left), (name, confidence)) in zip(boxes, names):
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 200, 100), 2)
                label = f"{name} ({confidence * 100:.1f}%)"
                cv2.putText(frame, label, (left, top - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 200, 100), 2)

            if frame_callback:
                frame_callback(frame, names)
            else:
                cv2.imshow("Face Recognition", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

    finally:
        reader.stop()
        picam2.stop()
        picam2.close()
        cv2.destroyAllWindows()
        print("[INFO] Recognition stopped cleanly.")
