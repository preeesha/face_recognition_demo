
#!/usr/bin/env python3
"""
Raspberry Pi FastAPI Server with Face Recognition Streaming
Handles dataset upload, database building, and live video streaming
"""

from fastapi import FastAPI, UploadFile, Form, BackgroundTasks
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
import shutil
import zipfile
from pathlib import Path
import cv2
import pickle
import numpy as np
from typing import Optional

# Import your custom modules
from pi_clean_image import FaceCleaner
from pi_build_database import PiDatabaseBuilder

# Import face recognition components
try:
    from insightface.app import FaceAnalysis
    INSIGHTFACE_AVAILABLE = True
except ImportError:
    INSIGHTFACE_AVAILABLE = False
    print("⚠️  InsightFace not available, will use Haar Cascade")

# ============================================================================
# CONFIGURATION
# ============================================================================

app = FastAPI(title="Raspberry Pi Face Recognition Server")

# Enable CORS for React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify your React app URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Paths
DATASET_DIR = Path("dataset")
CLEANED_DIR = Path("cleaned_dataset")
DB_PATH = Path("face_database.pkl")

# Global instances
face_cleaner = FaceCleaner()
face_recognizer = None  # Lazy loaded


# ============================================================================
# FACE RECOGNIZER CLASS
# ============================================================================

class LiveFaceRecognizer:
    """Simplified face recognizer for live streaming"""
    
    def __init__(self, db_path: str = "face_database.pkl", use_insightface: bool = True):
        """Initialize recognizer"""
        self.db_path = Path(db_path)
        self.users = {}
        self.use_insightface = use_insightface and INSIGHTFACE_AVAILABLE
        
        # Load database if exists
        if self.db_path.exists():
            with open(self.db_path, 'rb') as f:
                self.users = pickle.load(f)
            print(f"✅ Loaded database with {len(self.users)} users")
        else:
            print(f"⚠️  No database found at {db_path}")
        
        # Initialize face detector
        if self.use_insightface:
            self._init_insightface()
        else:
            self._init_haar()
    
    def _init_insightface(self):
        """Initialize InsightFace"""
        try:
            print("📊 Loading InsightFace...")
            self.app = FaceAnalysis(
                providers=['CPUExecutionProvider'],
                allowed_modules=['detection', 'recognition']
            )
            self.app.prepare(ctx_id=-1, det_size=(320, 320))
            print("✅ InsightFace ready")
        except Exception as e:
            print(f"⚠️  InsightFace failed: {e}, falling back to Haar")
            self._init_haar()
            self.use_insightface = False
    
    def _init_haar(self):
        """Initialize Haar Cascade"""
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        print("✅ Haar Cascade ready")
    
    def reload_database(self):
        """Reload database (call after building new database)"""
        if self.db_path.exists():
            with open(self.db_path, 'rb') as f:
                self.users = pickle.load(f)
            print(f"✅ Reloaded database with {len(self.users)} users")
            return True
        return False
    
    def get_faces_and_embeddings(self, image):
        """Extract all faces and their embeddings from image"""
        results = []
        
        if self.use_insightface:
            faces = self.app.get(image)
            for face in faces:
                bbox = face.bbox.astype(int)
                x, y, x2, y2 = bbox
                w, h = x2 - x, y2 - y
                results.append(((x, y, w, h), face.normed_embedding))
        else:
            # Haar + HOG fallback
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(gray, 1.2, 5, minSize=(50, 50))
            
            for (x, y, w, h) in faces:
                face_roi = gray[y:y+h, x:x+w]
                face_resized = cv2.resize(face_roi, (128, 128))
                
                # HOG features
                hog = cv2.HOGDescriptor(
                    _winSize=(128, 128),
                    _blockSize=(16, 16),
                    _blockStride=(8, 8),
                    _cellSize=(8, 8),
                    _nbins=9
                )
                features = hog.compute(face_resized).flatten()
                features = features / (np.linalg.norm(features) + 1e-7)
                
                results.append(((x, y, w, h), features))
        
        return results
    
    def recognize_embedding(self, embedding, threshold: float = 0.6):
        """Recognize face from embedding"""
        if embedding is None or not self.users:
            return None, 0
        
        best_match = None
        best_similarity = -1
        
        for name, data in self.users.items():
            for emb in data['embeddings']:
                similarity = np.dot(embedding, emb)
                
                if similarity > best_similarity:
                    best_similarity = similarity
                    best_match = name
        
        if best_similarity >= threshold:
            return best_match, best_similarity
        else:
            return None, best_similarity
    
    def draw_face_box(self, frame, bbox, name, confidence):
        """Draw bounding box and label on frame"""
        x, y, w, h = bbox
        
        # Choose color
        if name and name != "Unknown":
            color = (0, 255, 0)  # Green for recognized
        else:
            color = (0, 0, 255)  # Red for unknown
        
        # Draw rectangle
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        
        # Prepare label
        if name and name != "Unknown":
            label = f"{name} ({confidence:.2f})"
        else:
            label = f"Unknown ({confidence:.2f})"
        
        # Draw label background
        label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        label_w, label_h = label_size
        
        label_y = max(y - 10, label_h + 10)
        cv2.rectangle(frame, 
                     (x, label_y - label_h - 10), 
                     (x + label_w + 10, label_y + 5), 
                     color, -1)
        
        cv2.putText(frame, label, 
                   (x + 5, label_y - 5), 
                   cv2.FONT_HERSHEY_SIMPLEX, 
                   0.6, 
                   (255, 255, 255), 
                   2)


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def get_recognizer():
    """Get or create face recognizer instance"""
    global face_recognizer
    if face_recognizer is None:
        face_recognizer = LiveFaceRecognizer(
            db_path=str(DB_PATH),
            use_insightface=INSIGHTFACE_AVAILABLE
        )
    return face_recognizer


def generate_frames(camera_id: int = 0, threshold: float = 0.6, fps: int = 15):
    """
    Generate video frames with face recognition
    
    Args:
        camera_id: Camera device ID
        threshold: Recognition threshold
        fps: Target frames per second
    """
    recognizer = get_recognizer()
    cap = cv2.VideoCapture(camera_id)
    
    # Set camera properties
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, fps)
    
    if not cap.isOpened():
        print(f"❌ Cannot open camera {camera_id}")
        return
    
    print(f"🎥 Camera stream started (threshold={threshold}, fps={fps})")
    
    try:
        frame_count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                print("⚠️  Failed to read frame")
                break
            
            frame_count += 1
            
            # Get all faces and embeddings
            faces_data = recognizer.get_faces_and_embeddings(frame)
            
            # Draw info overlay
            info_text = f"Faces: {len(faces_data)} | Threshold: {threshold:.2f} | Frame: {frame_count}"
            cv2.putText(frame, info_text, (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, 
                       (255, 255, 255), 2)
            
            # Process each detected face
            if faces_data:
                for bbox, embedding in faces_data:
                    name, confidence = recognizer.recognize_embedding(embedding, threshold)
                    recognizer.draw_face_box(frame, bbox, name, confidence)
            else:
                # Show "No face detected" message
                cv2.putText(frame, "No face detected", (10, 60), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, 
                           (0, 165, 255), 2)
            
            # Encode frame as JPEG
            ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
            if not ret:
                continue
            
            frame_bytes = buffer.tobytes()
            
            # Yield frame in MJPEG format
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
    
    except Exception as e:
        print(f"❌ Streaming error: {e}")
    
    finally:
        cap.release()
        print("🛑 Camera stream stopped")


# ============================================================================
# API ENDPOINTS
# ============================================================================

@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "status": "online",
        "service": "Raspberry Pi Face Recognition Server",
        "database_loaded": len(get_recognizer().users) > 0,
        "users_count": len(get_recognizer().users)
    }


@app.post("/upload_dataset")
async def upload_dataset(person_name: str = Form(...), file: UploadFile = None):
    """
    Receive zipped dataset from Node backend.
    Extract -> Clean -> Save to cleaned_dataset.
    """
    try:
        person_zip_path = DATASET_DIR / f"{person_name}.zip"
        DATASET_DIR.mkdir(exist_ok=True)
        
        # Save uploaded zip
        with open(person_zip_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # Extract zip
        extract_path = DATASET_DIR / person_name
        with zipfile.ZipFile(person_zip_path, "r") as zip_ref:
            zip_ref.extractall(extract_path)
        
        print(f"📦 Extracted dataset for {person_name}")
        
        # Clean and crop faces
        person_clean_path = CLEANED_DIR / person_name
        face_cleaner.clean_dataset(str(DATASET_DIR), str(CLEANED_DIR))
        
        print(f"✅ Cleaned dataset ready for {person_name} at {person_clean_path}")
        
        return JSONResponse({
            "status": "ok", 
            "cleaned_path": str(person_clean_path),
            "message": f"Dataset for {person_name} uploaded and cleaned successfully"
        })
    
    except Exception as e:
        print(f"❌ Upload failed: {e}")
        return JSONResponse({"error": str(e)}, status_code=500)


@app.post("/build_database")
async def build_database(background_tasks: BackgroundTasks, fast: bool = False):
    """
    Build the face embedding database from cleaned_dataset.
    """
    try:
        print("🚀 Starting database build process...")
        background_tasks.add_task(run_database_build, fast)
        
        return JSONResponse({
            "status": "started", 
            "message": "Database building in background. This may take a few minutes."
        })
    
    except Exception as e:
        print(f"❌ Error starting build: {e}")
        return JSONResponse({"error": str(e)}, status_code=500)


def run_database_build(fast: bool):
    """Run the PiDatabaseBuilder (background task)"""
    try:
        builder = PiDatabaseBuilder(
            db_path=str(DB_PATH),
            use_insightface=not fast
        )
        builder.build_from_directory(str(CLEANED_DIR), images_per_person=50)
        
        print("✅ Database build complete. Reloading recognizer...")
        
        # Reload database in recognizer
        recognizer = get_recognizer()
        recognizer.reload_database()
        
        print("✅ Face recognizer updated with new database")
    
    except Exception as e:
        print(f"❌ Database build failed: {e}")


@app.get("/video_feed")
async def video_feed(
    threshold: float = 0.6,
    camera_id: int = 0,
    fps: int = 15
):
    """
    Stream live video with face recognition
    
    Args:
        threshold: Recognition confidence threshold (0.0-1.0)
        camera_id: Camera device ID (default: 0)
        fps: Target frames per second (default: 15)
    
    Returns:
        MJPEG stream with bounding boxes and labels
    """
    return StreamingResponse(
        generate_frames(camera_id=camera_id, threshold=threshold, fps=fps),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )


@app.get("/database_info")
async def database_info():
    """Get information about loaded database"""
    recognizer = get_recognizer()
    
    users_info = {}
    for name, data in recognizer.users.items():
        users_info[name] = {
            "embeddings_count": len(data.get('embeddings', [])),
            "sample_image": data.get('sample_image', 'N/A')
        }
    
    return {
        "database_exists": DB_PATH.exists(),
        "users_count": len(recognizer.users),
        "users": users_info,
        "using_insightface": recognizer.use_insightface
    }


@app.post("/reload_database")
async def reload_database():
    """Manually reload database (useful after building new database)"""
    try:
        recognizer = get_recognizer()
        success = recognizer.reload_database()
        
        if success:
            return JSONResponse({
                "status": "ok",
                "users_count": len(recognizer.users),
                "message": "Database reloaded successfully"
            })
        else:
            return JSONResponse({
                "status": "error",
                "message": "Database file not found"
            }, status_code=404)
    
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


# ============================================================================
# RUN SERVER
# ============================================================================

if __name__ == "__main__":
    import uvicorn
    
    print("""
╔═══════════════════════════════════════════════════════════════╗
║     Raspberry Pi Face Recognition Server                     ║
║     - Dataset Upload & Cleaning                              ║
║     - Database Building                                      ║
║     - Live Video Streaming with Recognition                  ║
╚═══════════════════════════════════════════════════════════════╝
    """)
    
    # Create directories
    DATASET_DIR.mkdir(exist_ok=True)
    CLEANED_DIR.mkdir(exist_ok=True)
    
    # Run server
    uvicorn.run(
        app, 
        host="0.0.0.0", 
        port=8000,
        log_level="info"
    )
