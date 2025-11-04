#!/usr/bin/env python3
"""
Raspberry Pi FastAPI Server - FIXED DATABASE SCHEMA
Fixed: SQLite column names now match the data being inserted
"""

from fastapi import FastAPI, UploadFile, Form, BackgroundTasks
from fastapi.responses import JSONResponse, StreamingResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path
import shutil, zipfile, cv2, pickle, numpy as np
import threading, time, json, requests, sqlite3
from datetime import datetime
from contextlib import contextmanager
from typing import Dict, Optional
from pi_clean_image import FaceCleaner
from pi_build_database import PiDatabaseBuilder

# --------------------------------------------------------
# InsightFace detection (fallback to Haar if unavailable)
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
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"]
)

DATASET_DIR   = Path("dataset")
CLEANED_DIR   = Path("cleaned_dataset")
RECORDINGS_DIR= Path("recordings"); RECORDINGS_DIR.mkdir(exist_ok=True)
DB_PATH       = Path("face_database.pkl")
SQLITE_DB_PATH= Path("recordings_database.db")
EXPRESS_URL   = "http://192.168.1.236:8080/upload_recording"

face_cleaner  = FaceCleaner()
face_recognizer = None

recording_state = {"is_recording": False, "current_recording": None, "lock": threading.Lock()}
camera_state = {"active_streams": set(), "lock": threading.Lock()}

# ========================================================
# SQLite database - FIXED SCHEMA
# ========================================================
class RecordingsDB:
    def __init__(self, path="recordings_database.db"):
        self.path = path
        self._init_db()

    @contextmanager
    def conn(self):
        c = sqlite3.connect(self.path)
        c.row_factory = sqlite3.Row
        try:
            yield c
            c.commit()
        finally:
            c.close()

    def _init_db(self):
        """Initialize database with CORRECT column names"""
        with self.conn() as c:
            cur = c.cursor()
            
            # Drop old table if schema is wrong
            cur.execute("DROP TABLE IF EXISTS recordings")
            
            # Create table with CORRECT columns matching metadata
            cur.execute("""
                CREATE TABLE IF NOT EXISTS recordings(
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    filename TEXT NOT NULL,
                    filepath TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    actual_duration REAL,
                    actual_fps REAL,
                    frame_count INTEGER,
                    threshold REAL,
                    file_size_bytes INTEGER,
                    file_size_mb REAL,
                    resolution TEXT,
                    codec TEXT,
                    total_unique_persons INTEGER,
                    detected_persons TEXT,
                    person_detection_counts TEXT
                )
            """)
        print("✅ SQLite database initialized with correct schema")

    def insert(self, meta: Dict):
        """Insert recording with all metadata fields"""
        with self.conn() as c:
            c.execute("""
                INSERT INTO recordings(
                    filename, filepath, timestamp,
                     actual_duration,
                    actual_fps, frame_count,
                    threshold, file_size_bytes, file_size_mb,
                    resolution, codec, total_unique_persons,
                    detected_persons, person_detection_counts
                ) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            """, (
                meta["filename"],
                meta["filepath"],
                meta["timestamp"],
                meta.get("actual_duration", 0),
                meta.get("actual_fps", 0),
                meta.get("frame_count", 0),
                meta.get("threshold", 0.6),
                meta.get("file_size_bytes", 0),
                meta.get("file_size_mb", 0),
                meta.get("resolution", "640x480"),
                meta.get("codec", "XVID"),
                meta.get("total_unique_persons", 0),
                json.dumps(meta.get("detected_persons", [])),
                json.dumps(meta.get("person_detection_counts", {}))
            ))
            return c.lastrowid
    
    def get_all(self):
        """Get all recordings with parsed JSON fields"""
        with self.conn() as c:
            rows = c.execute("SELECT * FROM recordings ORDER BY id DESC").fetchall()
            result = []
            for row in rows:
                rec = dict(row)
                # Parse JSON fields
                try:
                    rec['detected_persons'] = json.loads(rec.get('detected_persons', '[]'))
                except:
                    rec['detected_persons'] = []
                try:
                    rec['person_detection_counts'] = json.loads(rec.get('person_detection_counts', '{}'))
                except:
                    rec['person_detection_counts'] = {}
                result.append(rec)
            return result
    
    def delete_by_filename(self, filename: str):
        with self.conn() as c:
            c.execute("DELETE FROM recordings WHERE filename = ?", (filename,))
            return c.rowcount > 0

recordings_db = RecordingsDB()

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
        self.app = FaceAnalysis(providers=['CPUExecutionProvider'],
                                allowed_modules=['detection', 'recognition'])
        self.app.prepare(ctx_id=-1, det_size=(320, 320))
        print("✅ InsightFace ready")

    def _init_haar(self):
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
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
            for (x, y, w, h) in self.face_cascade.detectMultiScale(g, 1.2, 5, minSize=(50, 50)):
                face = cv2.resize(g[y:y + h, x:x + w], (128, 128))
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
        cv2.putText(f, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

# ========================================================
# Recording session
# ========================================================
class RecordingSession:
    def __init__(self, duration=60, threshold=0.6, camera_id=0, fps=15):
        self.duration, self.threshold, self.cam, self.fps = duration, threshold, camera_id, fps
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.filename = f"recording_{ts}.avi"
        self.path = RECORDINGS_DIR / self.filename
        self.start = self.end = None
        self.frames = 0
        self.detected = set()
        self.person_counts = {}

    def start_recording(self, rec: LiveFaceRecognizer):
        cap = cv2.VideoCapture(self.cam)
        cap.set(cv2.CAP_PROP_FPS, self.fps)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        writer = cv2.VideoWriter(str(self.path),
                               cv2.VideoWriter_fourcc(*'XVID'),
                               self.fps, (w, h))
        
        self.start = time.time()
        print(f"🎬 Recording started: {self.filename}")
        
        try:
            while time.time() - self.start < self.duration:
                ok, f = cap.read()
                if not ok:
                    break
                self.frames += 1
                
                for bbox, emb in rec.get_faces_and_embeddings(f):
                    name, conf = rec.recognize(emb, self.threshold)
                    if name:
                        self.detected.add(name)
                        self.person_counts[name] = self.person_counts.get(name, 0) + 1
                    rec.draw(f, bbox, name, conf)
                
                el = int(time.time() - self.start)
                cv2.circle(f, (20, 20), 8, (0, 0, 255), -1)
                cv2.putText(f, f"REC {el}/{self.duration}s", (40, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                writer.write(f)
            
            self.end = time.time()
        finally:
            cap.release()
            writer.release()
            print(f"✅ Recording completed: {self.filename}")

    def meta(self):
        """Generate complete metadata with ALL fields"""
        size_bytes = self.path.stat().st_size if self.path.exists() else 0
        size_mb = size_bytes / (1024 * 1024)
        dur = self.end - self.start if self.end and self.start else 0
        fps = self.frames / dur if dur > 0 else 0
        
        return {
            "filename": self.filename,
            "filepath": str(self.path.resolve()),
            "timestamp": datetime.now().isoformat(),
            "actual_duration": round(dur, 2),
            "actual_fps": round(fps, 2),
            "frame_count": self.frames,
            "threshold": self.threshold,
            "detected_persons": list(self.detected),
            "person_detection_counts": self.person_counts,
            "total_unique_persons": len(self.detected),
            "file_size_bytes": size_bytes,
            "file_size_mb": round(size_mb, 2),
            "resolution": "640x480",
            "codec": "XVID"
        }

# ========================================================
# Helpers
# ========================================================
def get_recognizer():
    global face_recognizer
    if not face_recognizer:
        face_recognizer = LiveFaceRecognizer(str(DB_PATH), INSIGHTFACE_AVAILABLE)
    return face_recognizer

def generate_frames(camera_id: int = 0, threshold: float = 0.6, fps: int = 15, stream_id: str = None):
    """Generate video frames with face recognition"""
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
    
    print(f"🎥 Camera stream started (ID: {stream_id})")
    
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
            cv2.putText(frame, info_text, (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            if faces_data:
                for bbox, embedding in faces_data:
                    name, confidence = recognizer.recognize(embedding, threshold)
                    recognizer.draw(frame, bbox, name, confidence)
            
            ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
            if not ret:
                continue
            
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
    
    except Exception as e:
        print(f"❌ Streaming error: {e}")
    finally:
        cap.release()
        if stream_id:
            with camera_state["lock"]:
                camera_state["active_streams"].discard(stream_id)
        print(f"🛑 Camera stream stopped")

# ========================================================
# Recording background task
# ========================================================
def run_recording_task(duration, threshold, camera_id, fps):
    try:
        with recording_state["lock"]:
            if recording_state["is_recording"]:
                print("⚠️ Recording already in progress")
                return
            recording_state["is_recording"] = True
        
        rec = get_recognizer()
        sess = RecordingSession(duration, threshold, camera_id, fps)
        recording_state["current_recording"] = sess
        
        # Start recording
        sess.start_recording(rec)
        
        # Get metadata
        meta = sess.meta()
        
        # Save to SQLite
        try:
            recording_id = recordings_db.insert(meta)
            print(f"💾 Saved to database with ID: {recording_id}")
        except Exception as e:
            print(f"❌ Database save failed: {e}")
            import traceback
            traceback.print_exc()
        
        print("📊 Metadata:", json.dumps(meta, indent=2))

        # Upload to Express
        try:
            print(f"📤 Uploading to Express: {EXPRESS_URL}")
            files = {"file": open(meta["filepath"], "rb")}
            data = {"metadata": json.dumps(meta)}
            r = requests.post(EXPRESS_URL, files=files, data=data, timeout=60)
            print(f"📤 Upload response: {r.status_code} - {r.text}")
            
            if r.status_code == 200:
                print("✅ Successfully uploaded to Express server")
            else:
                print(f"⚠️ Express returned status {r.status_code}")
                
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

# ========================================================
# API endpoints
# ========================================================
@app.get("/")
def root():
    return {
        "status": "ok",
        "message": "Pi Face Recognition Server",
        "active_streams": len(camera_state["active_streams"]),
        "is_recording": recording_state["is_recording"]
    }

@app.post("/upload_dataset")
async def upload_dataset(person_name: str = Form(...), file: UploadFile = None):
    """Receive zipped dataset from Node backend"""
    try:
        person_zip_path = DATASET_DIR / f"{person_name}.zip"
        DATASET_DIR.mkdir(exist_ok=True)
        
        with open(person_zip_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        extract_path = DATASET_DIR / person_name
        with zipfile.ZipFile(person_zip_path, "r") as zip_ref:
            zip_ref.extractall(extract_path)
        
        print(f"📦 Extracted dataset for {person_name}")
        face_cleaner.clean_dataset(str(DATASET_DIR), str(CLEANED_DIR))
        
        return JSONResponse({
            "status": "ok",
            "message": f"Dataset for {person_name} uploaded and cleaned successfully"
        })
    except Exception as e:
        print(f"❌ Upload failed: {e}")
        return JSONResponse({"error": str(e)}, status_code=500)

@app.post("/build_database")
async def build_database(background_tasks: BackgroundTasks, fast: bool = False):
    """Build the face embedding database"""
    try:
        print("🚀 Starting database build...")
        background_tasks.add_task(run_database_build, fast)
        return JSONResponse({
            "status": "started",
            "message": "Database building in background"
        })
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
        users_info[name] = {"embeddings_count": len(data.get('embeddings', []))}
    
    return {
        "database_exists": DB_PATH.exists(),
        "users_count": len(recognizer.users),
        "users": users_info,
        "using_insightface": recognizer.use_insightface
    }

@app.post("/reload_database")
async def reload_database():
    try:
        recognizer = get_recognizer()
        success = recognizer.reload_database()
        if success:
            return JSONResponse({"status": "ok", "users_count": len(recognizer.users)})
        else:
            return JSONResponse({"status": "error", "message": "Database file not found"}, status_code=404)
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)

@app.get("/video_feed")
async def video_feed(threshold: float = 0.6, fps: int = 15):
    stream_id = f"stream_{int(time.time() * 1000)}"
    return StreamingResponse(
        generate_frames(camera_id=0, threshold=threshold, fps=fps, stream_id=stream_id),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )

@app.post("/stop_stream")
async def stop_stream():
    with camera_state["lock"]:
        count = len(camera_state["active_streams"])
        camera_state["active_streams"].clear()
    return {"status": "ok", "message": f"Stopped {count} stream(s)"}

@app.post("/start_recording")
async def start_recording(
    background_tasks: BackgroundTasks,
    duration: int = 60,
    threshold: float = 0.6,
    fps: int = 15
):
    with recording_state["lock"]:
        if recording_state["is_recording"]:
            return JSONResponse({"status": "error", "message": "Recording already in progress"}, status_code=400)
    
    background_tasks.add_task(run_recording_task, duration, threshold, 0, fps)
    return {"status": "started", "duration": duration, "threshold": threshold, "fps": fps}

@app.get("/recording_status")
async def recording_status():
    with recording_state["lock"]:
        is_recording = recording_state["is_recording"]
        session = recording_state["current_recording"]
        
        if is_recording and session:
            elapsed = time.time() - session.start if session.start else 0
            remaining = session.duration - elapsed
            
            return {
                "is_recording": True,
                "filename": session.filename,
                "elapsed_seconds": round(elapsed, 1),
                "remaining_seconds": round(remaining, 1),
                "frame_count": session.frames,
                "detected_persons": list(session.detected)
            }
        else:
            return {"is_recording": False}

@app.get("/recordings")
def list_recordings():
    try:
        recordings = recordings_db.get_all()
        return {"count": len(recordings), "recordings": recordings}
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)

@app.get("/recording/{filename}")
async def download_recording(filename: str):
    filepath = RECORDINGS_DIR / filename
    if not filepath.exists():
        return JSONResponse({"error": "Recording file not found"}, status_code=404)
    return FileResponse(path=filepath, media_type="video/x-msvideo", filename=filename)

@app.delete("/recording/{filename}")
async def delete_recording(filename: str):
    try:
        db_deleted = recordings_db.delete_by_filename(filename)
        video_path = RECORDINGS_DIR / filename
        file_deleted = False
        if video_path.exists():
            video_path.unlink()
            file_deleted = True
        
        if not db_deleted and not file_deleted:
            return JSONResponse({"error": "Recording not found"}, status_code=404)
        
        return {"status": "ok", "deleted_from_database": db_deleted, "deleted_file": file_deleted}
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)

# ========================================================
# Run
# ========================================================
if __name__ == "__main__":
    import uvicorn
    
    print("""
╔═══════════════════════════════════════════════════════════════╗
║  Raspberry Pi Face Recognition Server                         ║                            
╚═══════════════════════════════════════════════════════════════╝
    """)
    
    DATASET_DIR.mkdir(exist_ok=True)
    CLEANED_DIR.mkdir(exist_ok=True)
    
    uvicorn.run(app, host="0.0.0.0", port=8000)
