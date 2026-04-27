import cv2
import os
from picamera2 import Picamera2
import time
import sys


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


def capture_faces(person_name, mode="add", save_base="dataset", frame_callback=None, stop_flag=None):
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
            if stop_flag and stop_flag():
                break

            frame = picam2.capture_array()

            gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.05, 5, minSize=(50, 50))

            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 2)

            info = f"Photos: {img_counter} "
            cv2.putText(frame, info, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            if frame_callback:
                result = frame_callback(frame)
                if result == "capture":
                    if len(faces) > 0:
                        img_path = f"{person_folder}/image_{img_counter}.jpg"
                        cv2.imwrite(img_path, frame)
                        img_counter += 1
                elif result == "stop":
                    break
            else:
                cv2.imshow("Capture Faces", frame)
                key = cv2.waitKey(1) & 0xFF
                if key == 27:
                    break
                elif key == 32:
                    if len(faces) > 0:
                        img_path = f"{person_folder}/image_{img_counter}.jpg"
                        cv2.imwrite(img_path, frame)
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
