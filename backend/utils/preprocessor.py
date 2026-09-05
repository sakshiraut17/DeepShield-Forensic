import numpy as np
import cv2
from PIL import Image
import librosa
import torch
import torchvision.transforms as transforms
from sklearn.preprocessing import StandardScaler
import json

class DataPreprocessor:
    """Preprocess different types of media for analysis"""
    
    def __init__(self):
        self.image_size = (224, 224)
        self.audio_sample_rate = 16000
        self.video_fps = 30
        
        # Image transforms
        self.image_transform = transforms.Compose([
            transforms.Resize(self.image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
        # Audio feature scaler
        self.audio_scaler = StandardScaler()
        
    def preprocess_image(self, image_path=None, image_array=None):
        """Preprocess image for model input"""
        if image_path:
            image = Image.open(image_path).convert('RGB')
        elif image_array is not None:
            if isinstance(image_array, np.ndarray):
                image = Image.fromarray(image_array)
            else:
                image = image_array
        else:
            raise ValueError("Either image_path or image_array must be provided")
        
        # Apply transforms
        image_tensor = self.image_transform(image)
        
        # Add batch dimension
        image_tensor = image_tensor.unsqueeze(0)
        
        return {
            'tensor': image_tensor,
            'array': np.array(image),
            'size': image.size,
            'mode': image.mode
        }
    
    def preprocess_audio(self, audio_path=None, audio_array=None, sr=None):
        """Preprocess audio for model input"""
        if audio_path:
            audio, sr = librosa.load(audio_path, sr=self.audio_sample_rate)
        elif audio_array is not None:
            audio = audio_array
            if sr is None:
                sr = self.audio_sample_rate
        else:
            raise ValueError("Either audio_path or audio_array must be provided")
        
        # Normalize
        audio = audio / (np.max(np.abs(audio)) + 1e-8)
        
        # Extract features
        features = self._extract_audio_features(audio, sr)
        
        # Create mel-spectrogram
        mel_spec = librosa.feature.melspectrogram(
            y=audio, 
            sr=sr,
            n_mels=128,
            fmax=8000
        )
        
        # Convert to log scale
        log_mel = librosa.power_to_db(mel_spec)
        
        # Normalize
        log_mel = (log_mel - log_mel.mean()) / (log_mel.std() + 1e-8)
        
        # Pad or truncate to fixed size
        target_length = 128
        if log_mel.shape[1] < target_length:
            pad_width = target_length - log_mel.shape[1]
            log_mel = np.pad(log_mel, ((0, 0), (0, pad_width)), mode='constant')
        else:
            log_mel = log_mel[:, :target_length]
        
        # Add channel dimension
        mel_tensor = torch.FloatTensor(log_mel).unsqueeze(0).unsqueeze(0)
        
        return {
            'tensor': mel_tensor,
            'features': features,
            'audio': audio,
            'sample_rate': sr,
            'duration': len(audio) / sr
        }
    
    def preprocess_video(self, video_path, max_frames=300, frame_skip=2):
        """Preprocess video for model input"""
        cap = cv2.VideoCapture(video_path)
        
        frames = []
        frame_count = 0
        processed_count = 0
        
        while cap.isOpened() and processed_count < max_frames:
            ret, frame = cap.read()
            if not ret:
                break
            
            if frame_count % frame_skip == 0:
                # Resize frame
                frame_resized = cv2.resize(frame, self.image_size)
                
                # Convert BGR to RGB
                frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
                
                # Convert to tensor
                frame_pil = Image.fromarray(frame_rgb)
                frame_tensor = self.image_transform(frame_pil)
                
                frames.append(frame_tensor)
                processed_count += 1
            
            frame_count += 1
        
        cap.release()
        
        # Stack frames
        if frames:
            video_tensor = torch.stack(frames, dim=1)  # (C, T, H, W)
            video_tensor = video_tensor.unsqueeze(0)   # (1, C, T, H, W)
        else:
            video_tensor = torch.zeros(1, 3, 1, 224, 224)
        
        return {
            'tensor': video_tensor,
            'frame_count': len(frames),
            'total_frames': frame_count
        }
    
    def _extract_audio_features(self, audio, sr):
        """Extract comprehensive audio features"""
        features = {}
        
        # MFCCs
        mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
        features['mfcc_mean'] = np.mean(mfccs, axis=1).tolist()
        features['mfcc_std'] = np.std(mfccs, axis=1).tolist()
        
        # Spectral features
        spectral_centroids = librosa.feature.spectral_centroid(y=audio, sr=sr)[0]
        features['spectral_centroid_mean'] = float(np.mean(spectral_centroids))
        features['spectral_centroid_std'] = float(np.std(spectral_centroids))
        
        spectral_rolloff = librosa.feature.spectral_rolloff(y=audio, sr=sr)[0]
        features['spectral_rolloff_mean'] = float(np.mean(spectral_rolloff))
        
        # Zero crossing rate
        zcr = librosa.feature.zero_crossing_rate(audio)[0]
        features['zcr_mean'] = float(np.mean(zcr))
        
        # RMS energy
        rms = librosa.feature.rms(y=audio)[0]
        features['rms_mean'] = float(np.mean(rms))
        
        # Pitch
        pitches, magnitudes = librosa.piptrack(y=audio, sr=sr)
        pitch_values = pitches[pitches > 0]
        if len(pitch_values) > 0:
            features['pitch_mean'] = float(np.mean(pitch_values))
            features['pitch_std'] = float(np.std(pitch_values))
        else:
            features['pitch_mean'] = 0
            features['pitch_std'] = 0
        
        return features
    
    def normalize_features(self, features, feature_type='audio'):
        """Normalize features using pre-fitted scaler"""
        if feature_type == 'audio':
            # Convert features dict to array
            feature_array = self._dict_to_array(features)
            
            # Normalize
            if hasattr(self.audio_scaler, 'mean_'):
                normalized = self.audio_scaler.transform(feature_array.reshape(1, -1))
            else:
                normalized = feature_array
            
            return self._array_to_dict(normalized.flatten(), features.keys())
        
        return features
    
    def _dict_to_array(self, feature_dict):
        """Convert feature dict to array"""
        array = []
        for key, value in feature_dict.items():
            if isinstance(value, list):
                array.extend(value)
            else:
                array.append(value)
        return np.array(array)
    
    def _array_to_dict(self, array, keys):
        """Convert array back to dict"""
        result = {}
        idx = 0
        for key in keys:
            if key in ['mfcc_mean', 'mfcc_std']:
                result[key] = array[idx:idx+13].tolist()
                idx += 13
            else:
                result[key] = float(array[idx])
                idx += 1
        return result