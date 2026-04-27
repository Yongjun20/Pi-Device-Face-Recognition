import face_recognition
import pickle
import cv2
import os

DATASET_DIR = "dataset"
ENCODINGS_FILE = "encodings.pickle"


def load_existing_encodings():
    """Load existing encodings and trained file list from pickle."""
    if not os.path.exists(ENCODINGS_FILE):
        return [], [], set()

    try:
        with open(ENCODINGS_FILE, "rb") as f:
            data = pickle.load(f)
        encodings = data.get("encodings", [])
        names = data.get("names", [])
        trained_files = set(data.get("trained_files", []))
        return encodings, names, trained_files
    except Exception as e:
        print(f"[WARN] Could not load existing encodings: {e}")
        return [], [], set()


def train_model(progress_callback=None):
    print("[INFO] Loading existing encodings...")
    knownEncodings, knownNames, trained_files = load_existing_encodings()

    new_count = 0
    skipped_count = 0

    if not os.path.exists(DATASET_DIR):
        print("[ERROR] Dataset folder not found.")
        return 0

    persons = [
        d for d in os.listdir(DATASET_DIR)
        if os.path.isdir(os.path.join(DATASET_DIR, d))
    ]

    for name in persons:
        person_dir = os.path.join(DATASET_DIR, name)
        images = [
            f for f in os.listdir(person_dir)
            if f.lower().endswith(('.jpg', '.jpeg', '.png'))
        ]

        for img_name in images:
            # Build a unique file key: person/filename
            file_key = f"{name}/{img_name}"

            # Skip if already trained
            if file_key in trained_files:
                skipped_count += 1
                continue

            image_path = os.path.join(person_dir, img_name)
            image = cv2.imread(image_path)
            if image is None:
                print(f"[WARN] Could not read image: {image_path}")
                continue

            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            boxes = face_recognition.face_locations(rgb, model="hog")

            if len(boxes) != 1:
                print(f"[SKIP] {file_key} — expected 1 face, found {len(boxes)}")
                continue

            encodings = face_recognition.face_encodings(rgb, boxes)
            for encoding in encodings:
                knownEncodings.append(encoding)
                knownNames.append(name)
                trained_files.add(file_key)
                new_count += 1
                print(f"[INFO] Trained: {file_key}")

            if progress_callback:
                progress_callback(f"Trained: {file_key}")

    if new_count == 0:
        print("[INFO] No new images to train.")
        if progress_callback:
            progress_callback("No new images to train.")
        return 0

    print(f"[INFO] Serializing encodings... ({new_count} new, {skipped_count} skipped)")

    data = {
        "encodings": knownEncodings,
        "names": knownNames,
        "trained_files": list(trained_files)
    }

    with open(ENCODINGS_FILE, "wb") as f:
        pickle.dump(data, f)

    print(f"[INFO] Training complete. New faces trained: {new_count}")
    print(f"[INFO] Total faces in database: {len(knownEncodings)}")

    return new_count


if __name__ == "__main__":
    train_model()
