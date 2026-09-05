import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import time
import math
from collections import deque

class DeepfakeVideoDetector:
    """Real video deepfake detection using frame-by-frame analysis and temporal consistency"""
    
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Video Detector using device: {self.device}")
        
        # Frame processing settings
        self.max_frames = 300  # Maximum frames to process
        self.frame_skip = 2    # Process every nth frame
        self.batch_size = 16   # Frames per batch
        
        # Load face detection model
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        
        # Initialize frame buffer for temporal analysis
        self.frame_buffer = deque(maxlen=30)
        
        # Create video analysis model
        self.model = self._create_video_model()
        
        # Image transforms
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
    
    def _create_video_model(self):
        """Create 3D CNN for video analysis"""
        class VideoCNN3D(nn.Module):
            def __init__(self):
                super().__init__()
                # 3D CNN layers
                self.conv1 = nn.Conv3d(3, 64, kernel_size=3, padding=1)
                self.conv2 = nn.Conv3d(64, 128, kernel_size=3, padding=1)
                self.conv3 = nn.Conv3d(128, 256, kernel_size=3, padding=1)
                self.conv4 = nn.Conv3d(256, 512, kernel_size=3, padding=1)
                
                self.pool = nn.MaxPool3d(2)
                self.dropout = nn.Dropout3d(0.3)
                self.relu = nn.ReLU()
                
                # Calculate size after convolutions
                self._to_linear = None
                self._get_conv_output()
                
                # Fully connected layers
                self.fc1 = nn.Linear(self._to_linear, 1024)
                self.fc2 = nn.Linear(1024, 512)
                self.fc3 = nn.Linear(512, 256)
                self.fc4 = nn.Linear(256, 2)
                
                self.dropout_fc = nn.Dropout(0.5)
            
            def _get_conv_output(self):
                with torch.no_grad():
                    dummy = torch.zeros(1, 3, 16, 112, 112)  # (batch, channels, frames, height, width)
                    dummy = self.pool(self.relu(self.conv1(dummy)))
                    dummy = self.pool(self.relu(self.conv2(dummy)))
                    dummy = self.pool(self.relu(self.conv3(dummy)))
                    dummy = self.pool(self.relu(self.conv4(dummy)))
                    self._to_linear = dummy.view(1, -1).size(1)
            
            def forward(self, x):
                # x shape: (batch, channels, frames, height, width)
                x = self.pool(self.relu(self.conv1(x)))
                x = self.pool(self.relu(self.conv2(x)))
                x = self.pool(self.relu(self.conv3(x)))
                x = self.pool(self.relu(self.conv4(x)))
                x = self.dropout(x)
                
                x = x.view(x.size(0), -1)
                
                x = self.relu(self.fc1(x))
                x = self.dropout_fc(x)
                x = self.relu(self.fc2(x))
                x = self.dropout_fc(x)
                x = self.relu(self.fc3(x))
                x = self.fc4(x)
                
                return x
        
        return VideoCNN3D().to(self.device)
    
    def analyze(self, video_path):
        """Analyze video for deepfake indicators"""
        start_time = time.time()
        
        # Open video
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            return {
                'label': 'Error',
                'confidence': 0.5,
                'frame_count': 0,
                'temporal_features': {},
                'analysis_time': 0
            }
        
        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps if fps > 0 else 0
        
        # Process frames
        frames = []
        frame_scores = []
        face_consistency = []
        motion_vectors = []
        
        frame_count = 0
        processed_count = 0
        
        while cap.isOpened() and processed_count < self.max_frames:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Skip frames
            if frame_count % self.frame_skip == 0:
                # Analyze frame
                frame_result = self._analyze_frame(frame)
                frame_scores.append(frame_result['score'])
                
                # Detect faces
                faces = self._detect_faces(frame)
                if len(faces) > 0:
                    face_consistency.append(len(faces))
                
                # Calculate motion
                if len(self.frame_buffer) > 0:
                    motion = self._calculate_motion(self.frame_buffer[-1], frame)
                    motion_vectors.append(motion)
                
                # Store frame for temporal analysis
                self.frame_buffer.append(frame)
                
                # Prepare for 3D CNN
                if len(self.frame_buffer) >= 16:
                    clip = list(self.frame_buffer)[-16:]
                    clip_tensor = self._prepare_clip(clip)
                    frames.append(clip_tensor)
                
                processed_count += 1
            
            frame_count += 1
        
        cap.release()
        
        # Calculate overall statistics
        if len(frame_scores) > 0:
            avg_frame_score = np.mean(frame_scores)
            std_frame_score = np.std(frame_scores)
            
            # Temporal consistency
            temporal_consistency = self._analyze_temporal_consistency(frame_scores)
            
            # Face consistency
            face_consistency_score = self._analyze_face_consistency(face_consistency)
            
            # Motion analysis
            motion_score = self._analyze_motion_patterns(motion_vectors)
            
            # Get 3D CNN prediction if we have clips
            cnn_score = 0.5
            if len(frames) > 0:
                cnn_scores = []
                for i in range(0, len(frames), self.batch_size):
                    batch = frames[i:i+self.batch_size]
                    if batch:
                        batch_tensor = torch.cat(batch, dim=0).to(self.device)
                        with torch.no_grad():
                            outputs = self.model(batch_tensor)
                            probs = torch.softmax(outputs, dim=1)
                            cnn_scores.extend(probs[:, 1].cpu().numpy())
                
                cnn_score = np.mean(cnn_scores) if cnn_scores else 0.5
            
            # Ensemble all scores
            weights = {
                'frame_analysis': 0.25,
                'temporal_consistency': 0.25,
                'face_consistency': 0.2,
                'motion_patterns': 0.15,
                'cnn_3d': 0.15
            }
            
            final_score = (
                weights['frame_analysis'] * avg_frame_score +
                weights['temporal_consistency'] * temporal_consistency +
                weights['face_consistency'] * face_consistency_score +
                weights['motion_patterns'] * motion_score +
                weights['cnn_3d'] * cnn_score
            )
            
            # Determine label
            label = "Deepfake Video" if final_score > 0.5 else "Real Video"
            
            # Prepare temporal features
            temporal_features = {
                'fps': fps,
                'duration': duration,
                'total_frames': total_frames,
                'processed_frames': processed_count,
                'score_variance': float(std_frame_score),
                'temporal_consistency': float(temporal_consistency),
                'face_consistency': float(face_consistency_score),
                'motion_consistency': float(motion_score)
            }
            
        else:
            final_score = 0.5
            label = "Unknown"
            temporal_features = {}
        
        analysis_time = time.time() - start_time
        
        return {
            'label': label,
            'confidence': float(final_score),
            'frame_count': processed_count,
            'temporal_features': temporal_features,
            'analysis_time': round(analysis_time, 3)
        }
    
    def analyze_frames(self, video_path):
        """Analyze individual frames for detailed report"""
        cap = cv2.VideoCapture(video_path)
        frame_analysis = []
        
        frame_count = 0
        while cap.isOpened() and frame_count < 50:  # Limit to 50 frames
            ret, frame = cap.read()
            if not ret:
                break
            
            if frame_count % 5 == 0:  # Analyze every 5th frame
                result = self._analyze_frame(frame)
                faces = self._detect_faces(frame)
                
                frame_analysis.append({
                    'frame_number': frame_count,
                    'confidence': result['score'],
                    'face_count': len(faces),
                    'blur_score': result['blur'],
                    'noise_score': result['noise']
                })
            
            frame_count += 1
        
        cap.release()
        return frame_analysis
    
    def _analyze_frame(self, frame):
        """Analyze individual frame"""
        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Calculate various metrics
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Blur detection
        blur_score = cv2.Laplacian(gray, cv2.CV_64F).var()
        normalized_blur = min(blur_score / 1000, 1.0)
        
        # Noise estimation
        noise_score = self._estimate_noise(gray)
        
        # Color consistency
        color_consistency = self._check_color_consistency(frame_rgb)
        
        # Frequency analysis
        freq_score = self._frequency_analysis(gray)
        
        # Combine scores
        scores = [
            normalized_blur * 0.3,
            (1 - noise_score) * 0.3,  # Lower noise is better
            color_consistency * 0.2,
            freq_score * 0.2
        ]
        
        final_score = np.mean(scores)
        
        return {
            'score': final_score,
            'blur': float(blur_score),
            'noise': float(noise_score),
            'color_consistency': float(color_consistency)
        }
    
    def _detect_faces(self, frame):
        """Detect faces in frame"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30)
        )
        return faces
    
    def _estimate_noise(self, gray_frame):
        """Estimate noise level in frame"""
        # Apply median filter
        denoised = cv2.medianBlur(gray_frame, 5)
        
        # Calculate difference
        noise = cv2.absdiff(gray_frame, denoised)
        
        # Normalize
        noise_level = np.mean(noise) / 255.0
        return noise_level
    
    def _check_color_consistency(self, frame):
        """Check color consistency across frame"""
        # Split channels
        if len(frame.shape) == 3:
            r, g, b = frame[:,:,0], frame[:,:,1], frame[:,:,2]
            
            # Calculate channel correlations
            corr_rg = np.corrcoef(r.flatten(), g.flatten())[0,1]
            corr_rb = np.corrcoef(r.flatten(), b.flatten())[0,1]
            corr_gb = np.corrcoef(g.flatten(), b.flatten())[0,1]
            
            # Average correlation
            avg_corr = np.mean([abs(corr_rg), abs(corr_rb), abs(corr_gb)])
            
            return avg_corr
        return 0.5
    
    def _frequency_analysis(self, gray_frame):
        """Analyze frequency domain"""
        # Apply FFT
        f = np.fft.fft2(gray_frame)
        fshift = np.fft.fftshift(f)
        
        # Calculate magnitude spectrum
        magnitude = np.log(np.abs(fshift) + 1)
        
        # Split into low and high frequencies
        h, w = magnitude.shape
        center_h, center_w = h//2, w//2
        
        # Low frequency region (center)
        low_freq = magnitude[center_h-30:center_h+30, center_w-30:center_w+30]
        low_energy = np.mean(low_freq)
        
        # High frequency region (edges)
        high_freq = np.concatenate([
            magnitude[:50, :].flatten(),
            magnitude[-50:, :].flatten(),
            magnitude[:, :50].flatten(),
            magnitude[:, -50:].flatten()
        ])
        high_energy = np.mean(high_freq)
        
        # Ratio indicates sharpness/artifacts
        if high_energy > 0:
            ratio = low_energy / high_energy
            return min(ratio / 10, 1.0)
        return 0.5
    
    def _calculate_motion(self, frame1, frame2):
        """Calculate motion between frames"""
        # Convert to grayscale
        gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
        
        # Calculate optical flow
        flow = cv2.calcOpticalFlowFarneback(
            gray1, gray2, None,
            pyr_scale=0.5, levels=3, winsize=15,
            iterations=3, poly_n=5, poly_sigma=1.2,
            flags=0
        )
        
        # Calculate magnitude
        mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        
        return np.mean(mag)
    
    def _analyze_temporal_consistency(self, scores):
        """Analyze temporal consistency of frame scores"""
        if len(scores) < 2:
            return 0.5
        
        # Calculate smoothness of scores
        differences = np.abs(np.diff(scores))
        avg_diff = np.mean(differences)
        
        # Real videos should have smooth transitions
        consistency = 1 - min(avg_diff * 2, 1)
        return consistency
    
    def _analyze_face_consistency(self, face_counts):
        """Analyze consistency of face detection"""
        if len(face_counts) < 2:
            return 0.5
        
        # Face count should be consistent in real videos
        unique_counts = len(set(face_counts))
        if unique_counts == 1:
            return 0.9  # Very consistent
        elif unique_counts <= 2:
            return 0.7
        else:
            return 0.3
    
    def _analyze_motion_patterns(self, motion_vectors):
        """Analyze motion patterns for anomalies"""
        if len(motion_vectors) < 5:
            return 0.5
        
        # Calculate statistics
        mean_motion = np.mean(motion_vectors)
        std_motion = np.std(motion_vectors)
        
        # Unnatural motion patterns (too smooth or too erratic)
        if std_motion < 0.1:
            return 0.8  # Too smooth - suspicious
        elif std_motion > 1.0:
            return 0.7  # Too erratic - suspicious
        else:
            return 0.3  # Natural motion
    
    def _prepare_clip(self, frames):
        """Prepare clip for 3D CNN"""
        processed_frames = []
        
        for frame in frames:
            # Resize and normalize frame
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_pil = Image.fromarray(frame_rgb)
            frame_tensor = self.transform(frame_pil)
            processed_frames.append(frame_tensor)
        
        # Stack frames
        clip = torch.stack(processed_frames, dim=1)  # (C, T, H, W)
        clip = clip.unsqueeze(0)  # (1, C, T, H, W)
        
        return clip