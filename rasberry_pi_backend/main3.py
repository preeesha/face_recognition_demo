from fastapi import FastAPI, UploadFile, Form, BackgroundTasks
from fastapi.responses import JSONResponse, StreamingResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path
import shutil, zipfile, cv2, pickle, numpy as np
import threading, time, json, requests
from datetime import datetime
from typing import Dict, Optional
from pi_clean_image import FaceCleaner
from pi_build_database import PiDatabaseBuilder

# --------------------------------------------------------
# InsightFace detection
# --------------------------------------------------------
try:
    from insightface.app import FaceAnalysis

    INSIGHTFACE_AVAILABLE = True
except ImportError:
    INSIGHTFACE_AVAILABLE = False
    print("⚠️ InsightFace not available — using Haar Cascade")

# --------------------------------------------------------
# Configuration
# --------------------------------------------------------
app = FastAPI(title="Raspberry Pi Face Recognition Server")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Get script directory
SCRIPT_DIR = Path(__file__).parent.absolute()

# Paths
DATASET_DIR = SCRIPT_DIR / "dataset"
CLEANED_DIR = SCRIPT_DIR / "cleaned_dataset"
RECORDINGS_DIR = SCRIPT_DIR / "recordings"
DB_PATH = SCRIPT_DIR / "face_database.pkl"

# Create directories
DATASET_DIR.mkdir(exist_ok=True)
CLEANED_DIR.mkdir(exist_ok=True)
RECORDINGS_DIR.mkdir(exist_ok=True)

EXPRESS_URL = "http://192.168.1.236:8080/upload_recording"

print(f"📁 Working directory: {SCRIPT_DIR}")
print(f"📁 Recordings directory: {RECORDINGS_DIR}")

face_cleaner = FaceCleaner()
face_recognizer = None

recording_state = {
    "is_recording": False,
    "current_recording": None,
    "lock": threading.Lock(),
    "upload_complete": False,  # ← ADD THIS
    "upload_status": "idle",# ← ADD THIS (values: "idle", "uploading", "success", "failed")
}  
camera_state = {"active_streams": set(), "lock": threading.Lock()}


# ========================================================
# Face recognizer
# ========================================================
class LiveFaceRecognizer:
    def __init__(self, db_path="face_database.pkl", use_insightface=True):
        self.db_path = Path(db_path)
        self.use_insightface = use_insightface and INSIGHTFACE_AVAILABLE
        self.users = {}
        if self.db_path.exists():
            self.users = pickle.load(open(self.db_path, "rb"))
            print(f"✅ Loaded {len(self.users)} users")
        else:
            print("⚠️ No database found")

        if self.use_insightface:
            self._init_insight()
        else:
            self._init_haar()

    def _init_insight(self):
        self.app = FaceAnalysis(
            providers=["CPUExecutionProvider"],
            allowed_modules=["detection", "recognition"],
        )
        self.app.prepare(ctx_id=-1, det_size=(320, 320))
        print("✅ InsightFace ready")

    def _init_haar(self):
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )
        print("✅ Haar ready")

    def reload_database(self):
        if self.db_path.exists():
            self.users = pickle.load(open(self.db_path, "rb"))
            print(f"✅ Reloaded {len(self.users)} users")
            return True
        return False

    def get_faces_and_embeddings(self, img):
        res = []
        if self.use_insightface:
            for f in self.app.get(img):
                x, y, x2, y2 = f.bbox.astype(int)
                res.append(((x, y, x2 - x, y2 - y), f.normed_embedding))
        else:
            g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            for x, y, w, h in self.face_cascade.detectMultiScale(
                g, 1.2, 5, minSize=(50, 50)
            ):
                face = cv2.resize(g[y : y + h, x : x + w], (128, 128))
                hog = cv2.HOGDescriptor()
                feat = hog.compute(face).flatten()
                feat /= np.linalg.norm(feat) + 1e-7
                res.append(((x, y, w, h), feat))
        return res

    def recognize(self, emb, thr=0.6):
        if not self.users:
            return None, 0
        best, score = None, -1
        for n, d in self.users.items():
            for e in d["embeddings"]:
                s = np.dot(emb, e)
                if s > score:
                    score, best = s, n
        return (best, score) if score >= thr else (None, score)

    def draw(self, f, bbox, name, conf):
        x, y, w, h = bbox
        color = (0, 255, 0) if name else (0, 0, 255)
        cv2.rectangle(f, (x, y), (x + w, y + h), color, 2)
        label = (name or "Unknown") + f" ({conf:.2f})"
        cv2.putText(
            f, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2
        )


# ========================================================
# Recording session - FIXED DURATION TIMING
# ========================================================
class RecordingSession:
    def __init__(self, duration=60, threshold=0.6, camera_id=0, fps=3):
        self.duration = duration
        self.threshold = threshold
        self.cam = camera_id
        self.fps = fps

        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.filename = f"recording_{ts}.avi"
        self.path = RECORDINGS_DIR / self.filename

        self.start = None
        self.end = None
        self.frames = 0
        self.detected = set()
        self.person_counts = {}

    def start_recording(self, rec: LiveFaceRecognizer):
        """Record video with PRECISE duration timing"""
        cap = cv2.VideoCapture(self.cam)
        # Retry logic — handles temporary camera lock or delay
        retry = 0
        while not cap.isOpened() and retry < 5:
            print(f"⚠️ Camera not ready, retrying... ({retry + 1}/5)")
            time.sleep(1)
            cap = cv2.VideoCapture(self.cam)
            retry += 1

        if not cap.isOpened():
            raise RuntimeError(f"❌ Cannot open camera {self.cam}")

        # Set camera properties
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, self.fps)

        # Get actual dimensions
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        print(f"📹 Camera opened: {w}x{h} @ {self.fps} FPS")

        # Initialize video writer
        fourcc = cv2.VideoWriter_fourcc(*"XVID")
        writer = cv2.VideoWriter(str(self.path), fourcc, self.fps, (w, h))

        if not writer.isOpened():
            cap.release()
            raise RuntimeError("❌ Failed to initialize video writer")

        self.start = time.time()
        print(f"🎬 Recording started: {self.filename} for {self.duration} seconds")

        try:
            # FIXED: Record for EXACT duration
            while True:
                elapsed = time.time() - self.start

                # Check if we've reached the duration
                if elapsed >= self.duration:
                    print(f"⏱️ Duration reached: {elapsed:.2f}s / {self.duration}s")
                    break

                ret, frame = cap.read()
                if not ret:
                    print("⚠️ Failed to read frame, continuing...")
                    continue

                self.frames += 1

                # Process faces
                for bbox, emb in rec.get_faces_and_embeddings(frame):
                    name, conf = rec.recognize(emb, self.threshold)
                    if name:
                        self.detected.add(name)
                        self.person_counts[name] = self.person_counts.get(name, 0) + 1
                    rec.draw(frame, bbox, name, conf)

                # Add recording overlay
                elapsed_int = int(elapsed)
                remaining = self.duration - elapsed_int
                cv2.circle(frame, (20, 20), 8, (0, 0, 255), -1)
                cv2.putText(
                    frame,
                    f"REC {elapsed_int}s/{self.duration}s",
                    (40, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 0, 255),
                    2,
                )

                # Write frame
                writer.write(frame)

                # Small sleep to prevent CPU overload while maintaining timing
                time.sleep(0.001)

            self.end = time.time()
            actual_duration = self.end - self.start
            print(
                f"✅ Recording completed: {actual_duration:.2f}s (target: {self.duration}s)"
            )

        except Exception as e:
            print(f"❌ Recording error: {e}")
            raise
        finally:
            writer.release()
            cap.release()
            print(f"📹 Camera and writer released")

    def meta(self):
        """Generate metadata"""
        file_exists = self.path.exists()
        size_bytes = self.path.stat().st_size if file_exists else 0
        size_mb = size_bytes / (1024 * 1024)
        dur = self.end - self.start if self.end and self.start else 0
        fps = self.frames / dur if dur > 0 else 0

        metadata = {
            "filename": self.filename,
            "filepath": str(self.path.absolute()),
            "timestamp": datetime.now().isoformat(),
            "duration_seconds": self.duration,
            "actual_duration": round(dur, 2),
            "target_fps": self.fps,
            "actual_fps": round(fps, 2),
            "frame_count": self.frames,
            "threshold": self.threshold,
            "detected_persons": list(self.detected),
            "person_detection_counts": self.person_counts,
            "total_unique_persons": len(self.detected),
            "file_size_bytes": size_bytes,
            "file_size_mb": round(size_mb, 2),
            "resolution": "640x480",
            "codec": "XVID",
            "file_exists": file_exists,
        }

        return metadata


# ========================================================
# Helpers
# ========================================================
def get_recognizer():
    global face_recognizer
    if not face_recognizer:
        face_recognizer = LiveFaceRecognizer(str(DB_PATH), INSIGHTFACE_AVAILABLE)
    return face_recognizer


def generate_frames(
    camera_id: int = 0, threshold: float = 0.6, fps: int = 15, stream_id: str = None
):
    """Generate video frames for streaming"""
    recognizer = get_recognizer()
    cap = cv2.VideoCapture(camera_id)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, fps)

    if not cap.isOpened():
        print(f"❌ Cannot open camera {camera_id}")
        return

    if stream_id:
        with camera_state["lock"]:
            camera_state["active_streams"].add(stream_id)

    print(f"🎥 Stream started (ID: {stream_id})")

    try:
        frame_count = 0
        while True:
            if stream_id:
                with camera_state["lock"]:
                    if stream_id not in camera_state["active_streams"]:
                        break

            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1
            faces_data = recognizer.get_faces_and_embeddings(frame)

            info_text = f"Faces: {len(faces_data)} | Threshold: {threshold:.2f}"
            cv2.putText(
                frame,
                info_text,
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2,
            )

            if faces_data:
                for bbox, embedding in faces_data:
                    name, confidence = recognizer.recognize(embedding, threshold)
                    recognizer.draw(frame, bbox, name, confidence)

            ret, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
            if not ret:
                continue

            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n" + buffer.tobytes() + b"\r\n"
            )

    except Exception as e:
        print(f"❌ Stream error: {e}")
    finally:
        cap.release()
        if stream_id:
            with camera_state["lock"]:
                camera_state["active_streams"].discard(stream_id)
        print(f"🛑 Stream stopped (ID: {stream_id})")


# ========================================================
# Recording background task
# ========================================================
def run_recording_task(duration, threshold, camera_id, fps):
    """Background task for recording"""
    try:
        with recording_state["lock"]:
            if recording_state["is_recording"]:
                print("⚠️ Recording already in progress")
                return
            recording_state["is_recording"] = True

        print(f"🎬 Starting recording: {duration}s, threshold={threshold}, fps={fps}")

        rec = get_recognizer()
        sess = RecordingSession(duration, threshold, camera_id, fps)
        recording_state["current_recording"] = sess

        # Start recording (blocks for duration)
        sess.start_recording(rec)

        # Get metadata
        meta = sess.meta()

        print(f"📊 Recording complete:")
        print(f"   - File: {meta['filename']}")
        print(
            f"   - Duration: {meta['actual_duration']}s (target: {meta['duration_seconds']}s)"
        )
        print(f"   - Frames: {meta['frame_count']}")
        print(f"   - Size: {meta['file_size_mb']} MB")
        print(f"   - Persons: {meta['detected_persons']}")
        print(f"   - File exists: {meta['file_exists']}")

        # Upload to Express (NO SQLite!)
        if not meta["file_exists"]:
            print(f"❌ Recording file does not exist: {meta['filepath']}")
            return

        try:
            print(f"📤 Uploading to Express: {EXPRESS_URL}")

            with open(meta["filepath"], "rb") as f:
                files = {"file": f}
                data = {"metadata": json.dumps(meta)}
                r = requests.post(EXPRESS_URL, files=files, data=data, timeout=120)

            print(f"📤 Upload response: {r.status_code}")

            if r.status_code == 200:
                print(f"✅ Successfully uploaded to Express server")
                print(f"   Response: {r.text[:200]}")
                try:
                    if Path(meta["filepath"]).exists():
                        Path(meta["filepath"]).unlink()
                        print(f"🗑️ Deleted local recording file: {meta['filename']}")
                except Exception as cleanup_err:
                    print(f"⚠️ Failed to delete local file: {cleanup_err}")
            else:
                print(f"⚠️ Express returned status {r.status_code}: {r.text}")

        except Exception as e:
            print(f"❌ Upload to Express failed: {e}")
            import traceback

            traceback.print_exc()

    except Exception as e:
        print(f"❌ Recording task failed: {e}")
        import traceback

        traceback.print_exc()
    finally:
        with recording_state["lock"]:
            recording_state["is_recording"] = False
            recording_state["current_recording"] = None
        print("🏁 Recording task finished")


# ========================================================
# API endpoints
# ========================================================
@app.get("/")
def root():
    return {
        "status": "ok",
        "message": "Pi Face Recognition Server (No SQLite)",
        "active_streams": len(camera_state["active_streams"]),
        "is_recording": recording_state["is_recording"],
        "recordings_dir": str(RECORDINGS_DIR),
    }


@app.post("/upload_dataset")
async def upload_dataset(person_name: str = Form(...), file: UploadFile = None):
    """Upload dataset from Express"""
    try:
        person_zip_path = DATASET_DIR / f"{person_name}.zip"

        with open(person_zip_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        extract_path = DATASET_DIR / person_name
        with zipfile.ZipFile(person_zip_path, "r") as zip_ref:
            zip_ref.extractall(extract_path)

        print(f"📦 Extracted dataset for {person_name}")
        face_cleaner.clean_dataset(str(DATASET_DIR), str(CLEANED_DIR))

        return JSONResponse(
            {"status": "ok", "message": f"Dataset for {person_name} uploaded"}
        )
    except Exception as e:
        print(f"❌ Upload failed: {e}")
        return JSONResponse({"error": str(e)}, status_code=500)


@app.post("/build_database")
async def build_database(background_tasks: BackgroundTasks, fast: bool = False):
    """Build face database"""
    try:
        print("🚀 Starting database build...")
        background_tasks.add_task(run_database_build, fast)
        return JSONResponse({"status": "started", "message": "Database building"})
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


def run_database_build(fast: bool):
    try:
        builder = PiDatabaseBuilder(db_path=str(DB_PATH), use_insightface=not fast)
        builder.build_from_directory(str(CLEANED_DIR), images_per_person=50)
        recognizer = get_recognizer()
        recognizer.reload_database()
        print("✅ Database built and reloaded")
    except Exception as e:
        print(f"❌ Database build failed: {e}")


@app.get("/database_info")
async def database_info():
    recognizer = get_recognizer()
    users_info = {}
    for name, data in recognizer.users.items():
        users_info[name] = {"embeddings_count": len(data.get("embeddings", []))}

    return {
        "database_exists": DB_PATH.exists(),
        "users_count": len(recognizer.users),
        "users": users_info,
        "using_insightface": recognizer.use_insightface,
    }


@app.post("/reload_database")
async def reload_database():
    try:
        recognizer = get_recognizer()
        success = recognizer.reload_database()
        if success:
            return JSONResponse({"status": "ok", "users_count": len(recognizer.users)})
        else:
            return JSONResponse(
                {"status": "error", "message": "Database not found"}, status_code=404
            )
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


@app.get("/video_feed")
async def video_feed(threshold: float = 0.6, fps: int = 15):
    stream_id = f"stream_{int(time.time() * 1000)}"
    return StreamingResponse(
        generate_frames(camera_id=0, threshold=threshold, fps=fps, stream_id=stream_id),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )


@app.post("/stop_stream")
async def stop_stream():
    with camera_state["lock"]:
        count = len(camera_state["active_streams"])
        camera_state["active_streams"].clear()
    print(f"🛑 Stopped {count} stream(s)")
    return {"status": "ok", "message": f"Stopped {count} stream(s)"}


@app.post("/start_recording")
async def start_recording(
    background_tasks: BackgroundTasks,
    duration: int = 60,
    threshold: float = 0.6,
    fps: int = 15,
):
    """Start recording with EXACT duration"""
    with recording_state["lock"]:
        if recording_state["is_recording"]:
            return JSONResponse(
                {"status": "error", "message": "Recording in progress"}, status_code=400
            )

    print(
        f"📝 Recording request: duration={duration}s, threshold={threshold}, fps={fps}"
    )

    background_tasks.add_task(run_recording_task, duration, threshold, 0, fps)

    return {
        "status": "started",
        "duration": duration,
        "threshold": threshold,
        "fps": fps,
        "message": f"Recording will run for exactly {duration} seconds",
    }


@app.get("/recording_status")
async def recording_status():
    with recording_state["lock"]:
        is_recording = recording_state["is_recording"]
        session = recording_state["current_recording"]

        if is_recording and session and session.start:
            elapsed = time.time() - session.start
            remaining = max(0, session.duration - elapsed)

            return {
                "is_recording": True,
                "filename": session.filename,
                "elapsed_seconds": round(elapsed, 1),
                "remaining_seconds": round(remaining, 1),
                "frame_count": session.frames,
                "detected_persons": list(session.detected),
                "duration": session.duration,
            }
        else:
            return {"is_recording": False}


@app.get("/recordings")
def list_recordings():
    """List recordings from disk"""
    try:
        recordings = []
        for avi_file in RECORDINGS_DIR.glob("*.avi"):
            stat = avi_file.stat()
            recordings.append(
                {
                    "filename": avi_file.name,
                    "filepath": str(avi_file.absolute()),
                    "size_mb": round(stat.st_size / (1024 * 1024), 2),
                    "created": datetime.fromtimestamp(stat.st_ctime).isoformat(),
                }
            )

        recordings.sort(key=lambda x: x["created"], reverse=True)
        return {"count": len(recordings), "recordings": recordings}
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


@app.get("/recording/{filename}")
async def download_recording(filename: str):
    filepath = RECORDINGS_DIR / filename
    if not filepath.exists():
        return JSONResponse({"error": "File not found"}, status_code=404)
    return FileResponse(path=filepath, media_type="video/x-msvideo", filename=filename)


@app.delete("/recording/{filename}")
async def delete_recording(filename: str):
    try:
        video_path = RECORDINGS_DIR / filename
        if video_path.exists():
            video_path.unlink()
            return {"status": "ok", "deleted": str(video_path)}
        return JSONResponse({"error": "File not found"}, status_code=404)
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


# ========================================================
# Run
# ========================================================
if __name__ == "__main__":
    import uvicorn

    print(
        """
╔═══════════════════════════════════════════════════════════════╗
║  Raspberry Pi Face Recognition Server                        ║
║  - NO SQLite (sends directly to Express)                     ║
║  - FIXED recording duration timing                           ║
║  - Proper camera handling                                    ║
╚═══════════════════════════════════════════════════════════════╝
    """
    )

    uvicorn.run(app, host="0.0.0.0", port=8000)
