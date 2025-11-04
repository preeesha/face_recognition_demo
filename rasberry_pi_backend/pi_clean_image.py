# pi_clean_image.py
import cv2
import os
import shutil
from pathlib import Path
import numpy as np

try:
    from insightface.app import FaceAnalysis
    INSIGHTFACE_AVAILABLE = True
except ImportError:
    INSIGHTFACE_AVAILABLE = False


class FaceCleaner:
    def __init__(self, detector=None):
        """
        Initialize with preloaded detector (for performance).
        If none is passed, we create our own lightweight detector.
        """
        self.detector = detector or self._init_detector()
        self.detection_scale = 0.5  # same as capture settings

    def _init_detector(self):
        """Initialize face detector (InsightFace or Haar)."""
        if INSIGHTFACE_AVAILABLE:
            try:
                app = FaceAnalysis(
                    providers=['CPUExecutionProvider'],
                    allowed_modules=['detection']
                )
                app.prepare(ctx_id=-1, det_size=(320, 320))
                print("✅ InsightFace initialized for cleaning.")
                return app
            except Exception as e:
                print(f"⚠ InsightFace failed: {e}")
        print("📸 Using Haar Cascade for cleaning.")
        return cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )

    def _detect_faces(self, frame):
        """Detect faces and return list of (x, y, w, h)."""
        small = cv2.resize(frame, (0, 0), fx=self.detection_scale, fy=self.detection_scale)

        if INSIGHTFACE_AVAILABLE and isinstance(self.detector, FaceAnalysis):
            faces = self.detector.get(small)
            scale = 1.0 / self.detection_scale
            boxes = []
            for face in faces:
                x1, y1, x2, y2 = face.bbox.astype(int)
                boxes.append((int(x1 * scale), int(y1 * scale), int(x2 - x1), int(y2 - y1)))
            return boxes
        else:
            gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
            faces = self.detector.detectMultiScale(
                gray, scaleFactor=1.2, minNeighbors=4, minSize=(40, 40)
            )
            return [(int(x / self.detection_scale), int(y / self.detection_scale),
                     int(w / self.detection_scale), int(h / self.detection_scale))
                    for (x, y, w, h) in faces]

    def clean_dataset(self, input_dir: str, output_dir: str):
        """
        Process all raw images: detect face, crop, and save to cleaned folder.
        """
        input_path = Path(input_dir)
        output_path = Path(output_dir)
        if not input_path.exists():
            raise FileNotFoundError(f"Input dataset not found: {input_path}")

        if output_path.exists():
            shutil.rmtree(output_path)
        output_path.mkdir(parents=True, exist_ok=True)

        person_dirs = [d for d in input_path.iterdir() if d.is_dir()]
        if not person_dirs:
            print("⚠ No person folders found.")
            return

        for person_dir in person_dirs:
            cleaned_person_dir = output_path / person_dir.name
            cleaned_person_dir.mkdir(parents=True, exist_ok=True)
            print(f"🧼 Cleaning {person_dir.name} ...")

            for img_path in person_dir.glob("*.jpg"):
                img = cv2.imread(str(img_path))
                if img is None:
                    continue
                faces = self._detect_faces(img)
                if not faces:
                    continue

                # Pick the largest face
                x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
                pad = 60
                x1, y1 = max(0, x - pad), max(0, y - pad)
                x2, y2 = min(img.shape[1], x + w + pad), min(img.shape[0], y + h + pad)
                face_img = img[y1:y2, x1:x2]

                # Save cropped face
                save_path = cleaned_person_dir / img_path.name
                cv2.imwrite(str(save_path), face_img, [cv2.IMWRITE_JPEG_QUALITY, 90])

            print(f"✅ Cleaned faces saved for {person_dir.name}")

        print("🎯 All datasets cleaned successfully!")
