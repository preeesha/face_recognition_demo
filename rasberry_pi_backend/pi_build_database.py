#!/usr/bin/env python3
"""
Raspberry Pi 5 Face Database Builder
Optimized for ARM - processes dataset and creates face embeddings database
"""

import pickle
import cv2
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional
import json
import sys

try:
    from insightface.app import FaceAnalysis
    INSIGHTFACE_AVAILABLE = True
except ImportError:
    INSIGHTFACE_AVAILABLE = False

class FaceDatabase:
    """Pickle-based face database optimized for Pi"""
    
    def __init__(self, db_path: str = "face_database.pkl"):
        """
        Initialize database
        
        Args:
            db_path: Path to pickle database file
        """
        self.db_path = Path(db_path)
        self.users = {}  # user_name: {embeddings: [], metadata: {}}
        
        # Load existing database if it exists
        if self.db_path.exists():
            self.load()
            print(f"✓ Loaded existing database: {len(self.users)} users")
        else:
            print("✓ Creating new database")
    
    def add_user(self, name: str, embedding: np.ndarray, metadata: Optional[Dict] = None):
        """
        Add a face embedding for a user
        
        Args:
            name: User name
            embedding: Face embedding vector (512-dim for InsightFace)
            metadata: Optional metadata (source image, quality, etc.)
        """
        if name not in self.users:
            self.users[name] = {
                'embeddings': [],
                'metadata': [],
                'created_at': datetime.now().isoformat(),
                'updated_at': datetime.now().isoformat()
            }
        
        self.users[name]['embeddings'].append(embedding)
        self.users[name]['metadata'].append(metadata or {})
        self.users[name]['updated_at'] = datetime.now().isoformat()
    
    def delete_user(self, name: str) -> bool:
        """Delete a user from database"""
        if name in self.users:
            del self.users[name]
            return True
        return False
    
    def get_user(self, name: str) -> Optional[Dict]:
        """Get user data"""
        return self.users.get(name)
    
    def list_users(self) -> List[str]:
        """Get list of all user names"""
        return list(self.users.keys())
    
    def get_statistics(self) -> Dict:
        """Get database statistics"""
        stats = {
            'total_users': len(self.users),
            'total_embeddings': sum(len(u['embeddings']) for u in self.users.values()),
            'users': {}
        }
        
        for name, data in self.users.items():
            stats['users'][name] = {
                'num_embeddings': len(data['embeddings']),
                'created_at': data.get('created_at', 'unknown'),
                'updated_at': data.get('updated_at', 'unknown')
            }
        
        return stats
    
    def save(self):
        """Save database to pickle file"""
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(self.db_path, 'wb') as f:
            pickle.dump(self.users, f, protocol=pickle.HIGHEST_PROTOCOL)
        
        print(f"✓ Database saved: {self.db_path}")
        print(f"  Users: {len(self.users)}")
        print(f"  Total embeddings: {sum(len(u['embeddings']) for u in self.users.values())}")
    
    def load(self):
        """Load database from pickle file"""
        with open(self.db_path, 'rb') as f:
            self.users = pickle.load(f)
    
    def export_stats(self, output_path: str = "database_stats.json"):
        """Export statistics to JSON file"""
        stats = self.get_statistics()
        
        with open(output_path, 'w') as f:
            json.dump(stats, f, indent=2)
        
        print(f"✓ Statistics exported: {output_path}")


class PiDatabaseBuilder:
    """Build face database from dataset - Pi optimized"""
    
    def __init__(self, db_path: str = "face_database.pkl", use_insightface: bool = True):
        """
        Initialize builder
        
        Args:
            db_path: Path to save database
            use_insightface: Use InsightFace (better) or Haar Cascade (faster)
        """
        self.database = FaceDatabase(db_path)
        self.use_insightface = use_insightface and INSIGHTFACE_AVAILABLE
        
        if self.use_insightface:
            self._init_insightface()
        else:
            self._init_haar()
    
    def _init_insightface(self):
        """Initialize InsightFace with Pi-optimized settings"""
        if not INSIGHTFACE_AVAILABLE:
            print("⚠️  InsightFace not available, using Haar Cascade fallback")
            self._init_haar()
            self.use_insightface = False
            return
        
        try:
            print("\n📊 Loading InsightFace models (this may take a moment on Pi)...")
            self.app = FaceAnalysis(
                providers=['CPUExecutionProvider'],
                allowed_modules=['detection', 'recognition']  # Only what we need
            )
            # Pi-optimized: smaller detection size
            self.app.prepare(ctx_id=-1, det_size=(320, 320))  # Reduced from 640
            self.recognition_method = "InsightFace (Pi-optimized)"
            print("✓ InsightFace loaded!")
        except Exception as e:
            print(f"⚠️  InsightFace failed to load: {e}")
            print("   Falling back to Haar Cascade...")
            self._init_haar()
            self.use_insightface = False
    
    def _init_haar(self):
        """Initialize Haar Cascade (fast fallback)"""
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        # For Haar, we'll use a simple feature extractor (HOG or just store images)
        self.recognition_method = "Haar Cascade + HOG Features"
        print("✓ Haar Cascade loaded (fast mode)")
        print("   Note: InsightFace provides better accuracy but is optional")
    
    def process_image_insightface(self, image_path: Path) -> List[Dict]:
        """
        Process image with InsightFace (better quality)
        
        Args:
            image_path: Path to image file
            
        Returns:
            List of face data with embeddings
        """
        # Read image
        image = cv2.imread(str(image_path))
        if image is None:
            return []
        
        # Resize if image is too large (Pi optimization)
        max_size = 1024
        if max(image.shape[:2]) > max_size:
            scale = max_size / max(image.shape[:2])
            new_width = int(image.shape[1] * scale)
            new_height = int(image.shape[0] * scale)
            image = cv2.resize(image, (new_width, new_height))
        
        # Detect and extract faces
        try:
            faces = self.app.get(image)
        except Exception as e:
            print(f"      Error processing {image_path.name}: {e}")
            return []
        
        results = []
        for face in faces:
            results.append({
                'embedding': face.normed_embedding,  # 512-dim normalized
                'bbox': face.bbox.astype(int).tolist(),
                'confidence': float(face.det_score),
                'landmarks': face.kps.astype(int).tolist() if hasattr(face, 'kps') else []
            })
        
        return results
    
    def process_image_haar(self, image_path: Path) -> List[Dict]:
        """
        Process image with Haar Cascade + HOG (faster on Pi)
        
        Args:
            image_path: Path to image file
            
        Returns:
            List of face data with HOG features
        """
        # Read image
        image = cv2.imread(str(image_path))
        if image is None:
            return []
        
        # Resize if too large
        max_size = 800
        if max(image.shape[:2]) > max_size:
            scale = max_size / max(image.shape[:2])
            new_width = int(image.shape[1] * scale)
            new_height = int(image.shape[0] * scale)
            image = cv2.resize(image, (new_width, new_height))
        
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Detect faces
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.2,
            minNeighbors=5,
            minSize=(50, 50)
        )
        
        results = []
        for (x, y, w, h) in faces:
            # Extract face region
            face_roi = gray[y:y+h, x:x+w]
            
            # Resize to standard size
            face_resized = cv2.resize(face_roi, (128, 128))
            
            # Compute HOG features
            hog = cv2.HOGDescriptor(
                _winSize=(128, 128),
                _blockSize=(16, 16),
                _blockStride=(8, 8),
                _cellSize=(8, 8),
                _nbins=9
            )
            features = hog.compute(face_resized)
            
            # Normalize features
            features = features.flatten()
            features = features / (np.linalg.norm(features) + 1e-7)
            
            results.append({
                'embedding': features,  # HOG features as "embedding"
                'bbox': [x, y, x+w, y+h],
                'confidence': 0.85,  # Haar doesn't provide confidence
                'landmarks': []
            })
        
        return results
    
    def process_image(self, image_path: Path) -> List[Dict]:
        """Process image with appropriate method"""
        if self.use_insightface:
            return self.process_image_insightface(image_path)
        else:
            return self.process_image_haar(image_path)
    
    def build_from_directory(self, dataset_path: str, images_per_person: int = 50,
                            skip_failed: bool = True):
        """
        Build database from directory structure
        
        Expected structure:
            dataset/
                person1/
                    img1.jpg
                    img2.jpg
                person2/
                    img1.jpg
        
        Args:
            dataset_path: Path to dataset directory
            images_per_person: Maximum images to process per person
            skip_failed: Continue if some images fail
        """
        dataset_path = Path(dataset_path)
        
        if not dataset_path.exists():
            print(f"❌ Dataset not found: {dataset_path}")
            return
        
        # Get all person directories
        person_dirs = [d for d in dataset_path.iterdir() 
                      if d.is_dir() and not d.name.startswith('.')]
        
        if len(person_dirs) == 0:
            print(f"❌ No person directories found in {dataset_path}")
            return
        
        print(f"\n{'='*70}")
        print(f"🍓 Building Face Database (Raspberry Pi 5)")
        print(f"{'='*70}")
        print(f"Dataset: {dataset_path}")
        print(f"People: {len(person_dirs)}")
        print(f"Max images per person: {images_per_person}")
        print(f"Recognition method: {self.recognition_method}")
        
        total_processed = 0
        total_added = 0
        
        # Process each person
        for person_idx, person_dir in enumerate(sorted(person_dirs), 1):
            person_name = person_dir.name
            
            # Get all images
            image_files = []
            for ext in ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']:
                image_files.extend(list(person_dir.glob(ext)))
            
            # Filter out preview directories
            image_files = [f for f in image_files if 'preview' not in str(f)]
            
            if len(image_files) == 0:
                print(f"\n⚠️  [{person_idx}/{len(person_dirs)}] {person_name}: No images found")
                continue
            
            print(f"\n📸 [{person_idx}/{len(person_dirs)}] Processing: {person_name}")
            print(f"   Found {len(image_files)} images")
            
            # Limit number of images
            image_files = sorted(image_files)[:images_per_person]
            
            added_count = 0
            failed_count = 0
            
            for img_idx, img_path in enumerate(image_files, 1):
                total_processed += 1
                
                # Process image
                try:
                    face_list = self.process_image(img_path)
                except Exception as e:
                    print(f"   ❌ Error: {img_path.name} - {e}")
                    failed_count += 1
                    if not skip_failed:
                        raise
                    continue
                
                if not face_list:
                    failed_count += 1
                    print(f"   ⚠️  No face: {img_path.name}")
                    continue
                
                # Pick best face by confidence
                best_face = max(face_list, key=lambda f: f['confidence'])
                
                if len(face_list) > 1:
                    print(f"   ⚠️  Multiple faces in {img_path.name} - using best")
                
                # Add to database
                metadata = {
                    'source_image': str(img_path),
                    'confidence': best_face['confidence'],
                    'bbox': best_face['bbox'],
                    'processed_at': datetime.now().isoformat(),
                    'method': self.recognition_method
                }
                
                self.database.add_user(person_name, best_face['embedding'], metadata)
                total_added += 1
                added_count += 1
                
                # Progress indicator (every 10 images)
                if added_count % 10 == 0:
                    print(f"   ✓ Processed {added_count}/{len(image_files)} images...")
            
            print(f"   ✅ Added {added_count} embeddings for {person_name}")
            if failed_count > 0:
                print(f"   ⚠️  Failed: {failed_count} images")
        
        # Save database
        print(f"\n{'='*70}")
        print(f"💾 Saving Database")
        print(f"{'='*70}")
        self.database.save()
        
        # Export statistics
        stats_path = self.database.db_path.parent / "database_stats.json"
        self.database.export_stats(str(stats_path))
        
        # Summary
        print(f"\n{'='*70}")
        print(f"✅ BUILD COMPLETE!")
        print(f"{'='*70}")
        print(f"Total images processed: {total_processed}")
        print(f"Total embeddings added: {total_added}")
        print(f"Database saved to: {self.database.db_path}")
        print(f"Method: {self.recognition_method}")
        
        # Show statistics
        print(f"\n📊 Database Statistics:")
        stats = self.database.get_statistics()
        for name, data in sorted(stats['users'].items()):
            print(f"   {name:20s}: {data['num_embeddings']:3d} embeddings")
        
        print(f"\n💡 Next steps:")
        print(f"   - Test recognition: python3 pi_test_recognition.py")
        print(f"   - View stats: python3 pi_manage_database.py --stats")


def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Build face database from dataset (Pi optimized)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Build with InsightFace (best quality, slower)
  python3 pi_build_database.py --dataset dataset
  
  # Build with Haar Cascade (faster)
  python3 pi_build_database.py --dataset dataset --fast
  
  # Custom output location
  python3 pi_build_database.py --dataset dataset --output models/faces.pkl
  
  # Process more images per person
  python3 pi_build_database.py --dataset dataset --images 100
        """
    )
    
    parser.add_argument(
        '--dataset',
        type=str,
        default='dataset',
        help='Path to dataset directory (default: dataset)'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        default='face_database.pkl',
        help='Output database file (default: face_database.pkl)'
    )
    
    parser.add_argument(
        '--images',
        type=int,
        default=50,
        help='Max images per person (default: 50)'
    )
    
    parser.add_argument(
        '--fast',
        action='store_true',
        help='Use Haar Cascade instead of InsightFace (faster but less accurate)'
    )
    
    parser.add_argument(
        '--strict',
        action='store_true',
        help='Stop on first error (default: skip failed images)'
    )
    
    args = parser.parse_args()
    
    print("""
╔═══════════════════════════════════════════════════════════════╗
║         Raspberry Pi 5 Face Database Builder                 ║
║         Optimized for ARM Performance                        ║
╚═══════════════════════════════════════════════════════════════╝
    """)
    
    # Check dataset exists
    if not Path(args.dataset).exists():
        print(f"❌ Dataset not found: {args.dataset}")
        print("\nPlease capture dataset first:")
        print(f"  python3 pi_face_capture.py")
        return
    
    # Check for empty dataset
    person_dirs = [d for d in Path(args.dataset).iterdir() 
                  if d.is_dir() and not d.name.startswith('.')]
    if len(person_dirs) == 0:
        print(f"❌ No people found in dataset")
        print("\nCapture some faces first:")
        print(f"  python3 pi_face_capture.py --name yourname")
        return
    
    # Build database
    try:
        builder = PiDatabaseBuilder(
            db_path=args.output,
            use_insightface=not args.fast
        )
        
        builder.build_from_directory(
            dataset_path=args.dataset,
            images_per_person=args.images,
            skip_failed=not args.strict
        )
        
        print("\n🎉 Database ready!")
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Build interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()