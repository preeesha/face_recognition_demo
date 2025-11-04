#!/usr/bin/env python3
"""
Raspberry Pi 5 Face Recognition Test
Quick test to verify database works with GUI and headless modes
"""

import pickle
import cv2
import numpy as np
from pathlib import Path
import sys

try:
    from insightface.app import FaceAnalysis
    INSIGHTFACE_AVAILABLE = True
except ImportError:
    INSIGHTFACE_AVAILABLE = False

class SimpleFaceRecognizer:
    """Simple face recognition using cosine similarity"""
    
    def __init__(self, db_path: str = "face_database.pkl", 
                 use_insightface: bool = True):
        """Initialize recognizer"""
        self.db_path = Path(db_path)
        
        # Load database
        if not self.db_path.exists():
            print(f"❌ Database not found: {db_path}")
            print("\nBuild database first:")
            print("  python3 pi_build_database.py --dataset dataset")
            sys.exit(1)
        
        with open(self.db_path, 'rb') as f:
            self.users = pickle.load(f)
        
        print(f"✓ Loaded database with {len(self.users)} users")
        
        # Initialize face detector
        self.use_insightface = use_insightface and INSIGHTFACE_AVAILABLE
        
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
            print("✓ InsightFace ready")
        except Exception as e:
            print(f"⚠️  InsightFace failed: {e}")
            self._init_haar()
            self.use_insightface = False
    
    def _init_haar(self):
        """Initialize Haar Cascade"""
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        print("✓ Haar Cascade ready (fast mode)")
    
    def get_faces_and_embeddings(self, image):
        """
        Extract all faces and their embeddings from image
        
        Returns:
            List of tuples: [(bbox, embedding), ...]
            bbox format: (x, y, w, h)
        """
        results = []
        
        if self.use_insightface:
            faces = self.app.get(image)
            for face in faces:
                # Convert InsightFace bbox to (x, y, w, h)
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
    
    def get_embedding(self, image):
        """Extract embedding from image (single face)"""
        if self.use_insightface:
            faces = self.app.get(image)
            if not faces:
                return None
            return faces[0].normed_embedding
        else:
            # Haar + HOG fallback
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(gray, 1.2, 5, minSize=(50, 50))
            
            if len(faces) == 0:
                return None
            
            # Get first face
            (x, y, w, h) = faces[0]
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
            
            return features
    
    def recognize_embedding(self, embedding, threshold: float = 0.6):
        """
        Recognize a face from its embedding
        
        Args:
            embedding: Face embedding
            threshold: Similarity threshold
            
        Returns:
            (name, confidence) or (None, confidence)
        """
        if embedding is None:
            return None, 0
        
        # Compare with all users
        best_match = None
        best_similarity = -1
        
        for name, data in self.users.items():
            # Compare with all embeddings for this user
            for emb in data['embeddings']:
                # Cosine similarity
                similarity = np.dot(embedding, emb)
                
                if similarity > best_similarity:
                    best_similarity = similarity
                    best_match = name
        
        # Check threshold
        if best_similarity >= threshold:
            return best_match, best_similarity
        else:
            return None, best_similarity
    
    def recognize(self, image, threshold: float = 0.6):
        """
        Recognize face in image
        
        Args:
            image: Input image
            threshold: Similarity threshold (higher = stricter)
            
        Returns:
            (name, confidence) or (None, 0) if no match
        """
        # Get embedding
        test_embedding = self.get_embedding(image)
        return self.recognize_embedding(test_embedding, threshold)
    
    def draw_face_box(self, frame, bbox, name, confidence, color=None):
        """
        Draw bounding box and label on frame
        
        Args:
            frame: Image frame
            bbox: Bounding box (x, y, w, h)
            name: Person's name or "Unknown"
            confidence: Recognition confidence
            color: Box color (B, G, R)
        """
        x, y, w, h = bbox
        
        # Choose color based on recognition
        if color is None:
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
        
        # Position label above the box
        label_y = max(y - 10, label_h + 10)
        cv2.rectangle(frame, 
                     (x, label_y - label_h - 10), 
                     (x + label_w + 10, label_y + 5), 
                     color, -1)
        
        # Draw text
        cv2.putText(frame, label, 
                   (x + 5, label_y - 5), 
                   cv2.FONT_HERSHEY_SIMPLEX, 
                   0.6, 
                   (255, 255, 255), 
                   2)
    
    def test_from_camera(self, camera_id: int = 0, threshold: float = 0.6, 
                        gui: bool = False):
        """
        Test recognition from camera
        
        Args:
            camera_id: Camera device ID
            threshold: Recognition threshold
            gui: Show GUI window (True) or headless mode (False)
        """
        print(f"\n{'='*70}")
        print(f"📷 Testing Face Recognition from Camera")
        print(f"{'='*70}")
        print(f"Mode: {'GUI' if gui else 'Headless'}")
        print(f"Threshold: {threshold}")
        
        if gui:
            print(f"Press 'q' to quit, 's' to save screenshot")
        else:
            print(f"Press Ctrl+C to stop")
        
        cap = cv2.VideoCapture(camera_id)
        
        if not cap.isOpened():
            print(f"❌ Cannot open camera {camera_id}")
            return
        
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        if gui:
            print("\n🎬 Camera started with GUI")
            window_name = "Face Recognition - Press 'q' to quit"
            cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        else:
            print("\n🎬 Camera started (headless - no window)")
            print("   Recognition results will be printed here")
        
        frame_count = 0
        screenshot_count = 0
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    continue
                
                frame_count += 1
                
                # Process every frame for GUI, every 30 frames for headless
                process_frame = gui or (frame_count % 30 == 0)
                
                if process_frame:
                    # Get all faces and embeddings
                    faces_data = self.get_faces_and_embeddings(frame)
                    
                    if gui:
                        # Draw frame info
                        info_text = f"Faces: {len(faces_data)} | Threshold: {threshold:.2f}"
                        cv2.putText(frame, info_text, (10, 30), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, 
                                   (255, 255, 255), 2)
                    
                    if faces_data:
                        for bbox, embedding in faces_data:
                            # Recognize face
                            name, confidence = self.recognize_embedding(embedding, threshold)
                            
                            if not gui:
                                # Print results in headless mode
                                if name:
                                    print(f"   ✓ Recognized: {name} (confidence: {confidence:.3f})")
                                else:
                                    print(f"   ⚠️  Unknown face (best match: {confidence:.3f})")
                            else:
                                # Draw bounding box in GUI mode
                                self.draw_face_box(frame, bbox, name, confidence)
                    else:
                        if not gui and frame_count % 30 == 0:
                            print(f"   ⏸  No face detected")
                
                # Show frame if GUI mode
                if gui:
                    cv2.imshow(window_name, frame)
                    
                    # Handle key presses
                    key = cv2.waitKey(1) & 0xFF
                    
                    if key == ord('q'):
                        print("\n⚠️  Quit requested")
                        break
                    elif key == ord('s'):
                        # Save screenshot
                        screenshot_count += 1
                        filename = f"screenshot_{screenshot_count:03d}.jpg"
                        cv2.imwrite(filename, frame)
                        print(f"📸 Screenshot saved: {filename}")
        
        except KeyboardInterrupt:
            print("\n\n⚠️  Stopped by user")
        
        finally:
            cap.release()
            if gui:
                cv2.destroyAllWindows()
    
    def test_from_image(self, image_path: str, threshold: float = 0.6, 
                       gui: bool = False, save_output: bool = False):
        """
        Test recognition from image file
        
        Args:
            image_path: Path to image file
            threshold: Recognition threshold
            gui: Show GUI window
            save_output: Save annotated image
        """
        print(f"\n{'='*70}")
        print(f"🖼️  Testing Face Recognition from Image")
        print(f"{'='*70}")
        print(f"Image: {image_path}")
        print(f"Threshold: {threshold}")
        
        # Load image
        image = cv2.imread(image_path)
        
        if image is None:
            print(f"❌ Cannot load image: {image_path}")
            return
        
        # Get all faces
        faces_data = self.get_faces_and_embeddings(image)
        
        print(f"\nFound {len(faces_data)} face(s)")
        
        if not faces_data:
            print("❌ No faces detected")
            return
        
        # Process each face
        for i, (bbox, embedding) in enumerate(faces_data, 1):
            name, confidence = self.recognize_embedding(embedding, threshold)
            
            print(f"\nFace {i}:")
            if name:
                print(f"  ✅ Recognized: {name}")
                print(f"     Confidence: {confidence:.3f}")
            else:
                print(f"  ❌ Unknown face")
                print(f"     Best match: {confidence:.3f} (below threshold {threshold})")
            
            # Draw bounding box if GUI or save mode
            if gui or save_output:
                self.draw_face_box(image, bbox, name, confidence)
        
        # Save output
        if save_output:
            output_path = image_path.replace('.jpg', '_annotated.jpg')
            output_path = output_path.replace('.png', '_annotated.png')
            if output_path == image_path:
                output_path = 'annotated_output.jpg'
            
            cv2.imwrite(output_path, image)
            print(f"\n💾 Annotated image saved: {output_path}")
        
        # Show GUI
        if gui:
            window_name = "Face Recognition Result - Press any key to close"
            cv2.imshow(window_name, image)
            print("\n👁️  Image displayed. Press any key to close...")
            cv2.waitKey(0)
            cv2.destroyAllWindows()


def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Test face recognition (Pi optimized with GUI support)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Test from camera with GUI
  python3 pi_test_recognition.py --camera --gui
  
  # Test from camera headless (original mode)
  python3 pi_test_recognition.py --camera
  
  # Test from image with GUI
  python3 pi_test_recognition.py --image test.jpg --gui
  
  # Test from image and save annotated output
  python3 pi_test_recognition.py --image test.jpg --save
  
  # Adjust threshold with GUI
  python3 pi_test_recognition.py --camera --gui --threshold 0.7
  
  # Use fast mode (Haar Cascade) with GUI
  python3 pi_test_recognition.py --camera --gui --fast
        """
    )
    
    parser.add_argument(
        '--database',
        type=str,
        default='face_database.pkl',
        help='Database file (default: face_database.pkl)'
    )
    
    parser.add_argument(
        '--camera',
        action='store_true',
        help='Test from camera'
    )
    
    parser.add_argument(
        '--image',
        type=str,
        help='Test from image file'
    )
    
    parser.add_argument(
        '--threshold',
        type=float,
        default=0.6,
        help='Recognition threshold (default: 0.6)'
    )
    
    parser.add_argument(
        '--fast',
        action='store_true',
        help='Use Haar Cascade (faster, less accurate)'
    )
    
    parser.add_argument(
        '--gui',
        action='store_true',
        help='Show GUI window with bounding boxes and labels'
    )
    
    parser.add_argument(
        '--save',
        action='store_true',
        help='Save annotated image (for --image mode)'
    )
    
    args = parser.parse_args()
    
    print("""
╔═══════════════════════════════════════════════════════════════╗
║         Raspberry Pi 5 Face Recognition Test                 ║
║              with GUI and Bounding Box Support                ║
╚═══════════════════════════════════════════════════════════════╝
    """)
    
    # Initialize recognizer
    try:
        recognizer = SimpleFaceRecognizer(
            db_path=args.database,
            use_insightface=not args.fast
        )
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)
    
    # Run test
    if args.camera:
        recognizer.test_from_camera(threshold=args.threshold, gui=args.gui)
    elif args.image:
        recognizer.test_from_image(args.image, threshold=args.threshold, 
                                   gui=args.gui, save_output=args.save)
    else:
        print("\n❌ Please specify --camera or --image")
        print("   See --help for examples")


if __name__ == "__main__":
    main()
