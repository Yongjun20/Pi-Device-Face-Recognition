import cv2
import os
from picamera2 import Picamera2
import time
import sys

AUTO_LIMIT    = 10   # max photos in auto mode
AUTO_COOLDOWN = 2    # seconds between auto-captures


def open_camera_with_retry(retries=5, delay=1):
    for i in range(retries):
        try:
            cam = Picamera2()
            return cam
        except RuntimeError:
            print(f"[WARN] Camera busy, retrying ({i+1}/{retries})...")
            time.sleep(delay)
    raise RuntimeError("Camera is busy. Please close other apps and retry.")


def count_images_in_folder(folder_path):
    if not os.path.exists(folder_path):
        return 0
    return len([
        f for f in os.listdir(folder_path)
        if f.lower().endswith(('.jpg', '.jpeg', '.png'))
    ])


def capture_faces(person_name, mode="add", save_base="dataset",
                  frame_callback=None, stop_flag=None, auto_capture=False):

    cascade_path = "/usr/share/opencv4/haarcascades/haarcascade_frontalface_default.xml"
    face_cascade = cv2.CascadeClassifier(cascade_path)
    if face_cascade.empty():
        raise RuntimeError("Face cascade not loaded")

    person_folder = f"dataset/{person_name.lower()}"
    os.makedirs(person_folder, exist_ok=True)

    existing_count = count_images_in_folder(person_folder)

    if mode == "replace" and existing_count > 0:
        for f in os.listdir(person_folder):
            if f.lower().endswith(('.jpg', '.jpeg', '.png')):
                os.remove(os.path.join(person_folder, f))
        img_counter = 0
    else:
        img_counter = existing_count

    # Auto-capture state
    auto_count     = 0
    last_auto_time = 0.0

    picam2 = None
    try:
        picam2 = open_camera_with_retry()
        picam2.configure(
            picam2.create_preview_configuration(
                main={"format": 'XRGB8888', "size": (640, 480)}
            )
        )
        picam2.start()
        time.sleep(0.5)

        start_counter = img_counter

        while True:
            # ── Stop flag ──────────────────────────────────────
            if stop_flag and stop_flag():
                break

            frame = picam2.capture_array()

            # Detect faces — use grayscale for detection only
            gray  = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.05, 5, minSize=(50, 50))

            # ── Draw face boxes ────────────────────────────────
            for (x, y, w, h) in faces:
                # Green box = exactly 1 face (good for capture)
                # Yellow box = multiple faces detected
                color = (0, 255, 0) if len(faces) == 1 else (0, 255, 255)
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

            # ── Overlay text ───────────────────────────────────
            if auto_capture:
                info = f"Photos: {img_counter}  |  Auto: {auto_count}/{AUTO_LIMIT}"
            else:
                info = f"Photos: {img_counter}"

            cv2.putText(frame, info, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            # ── Auto-capture logic ─────────────────────────────
            if auto_capture:
                now = time.time()
                if (
                    len(faces) == 1                        # exactly 1 face
                    and auto_count < AUTO_LIMIT            # under 10 photos
                    and (now - last_auto_time) >= AUTO_COOLDOWN  # cooldown passed
                ):
                    # Save as BGR (cv2.imwrite expects BGR, PiCamera gives RGB)
                    save_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                    img_path   = f"{person_folder}/image_{img_counter}.jpg"
                    cv2.imwrite(img_path, save_frame)
                    img_counter   += 1
                    auto_count    += 1
                    last_auto_time = now

                # Auto stop after 10 photos
                if auto_count >= AUTO_LIMIT:
                    if frame_callback:
                        frame_callback(frame)   # show last frame before stopping
                    break

            # ── Frame callback (GUI mode) ──────────────────────
            if frame_callback:
                result = frame_callback(frame)
                if result == "capture":
                    # Manual capture — only save if exactly 1 face detected
                    if len(faces) == 1:
                        save_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                        img_path   = f"{person_folder}/image_{img_counter}.jpg"
                        cv2.imwrite(img_path, save_frame)
                        img_counter += 1
                elif result == "stop":
                    break

            # ── Standalone mode (no GUI) ───────────────────────
            else:
                cv2.imshow("Capture Faces", frame)
                key = cv2.waitKey(1) & 0xFF
                if key == 27:   # Esc
                    break
                elif key == 32:  # Space
                    if len(faces) == 1:
                        save_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                        img_path   = f"{person_folder}/image_{img_counter}.jpg"
                        cv2.imwrite(img_path, save_frame)
                        img_counter += 1

    finally:
        if not frame_callback:
            cv2.destroyAllWindows()
        if picam2:
            try:
                picam2.stop()
            except:
                pass
            try:
                picam2.close()
            except:
                pass
            del picam2
            time.sleep(1)

    return img_counter - start_counter, img_counter


if __name__ == "__main__":
    name = sys.argv[1]
    mode = sys.argv[2] if len(sys.argv) > 2 else "add"
    capture_faces(name, mode)
