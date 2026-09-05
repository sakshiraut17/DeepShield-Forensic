import numpy as np
import cv2
from PIL import Image
import librosa
import scipy.signal
from scipy.fftpack import fft2, fftshift
import hashlib
import json

class FeatureExtractor:
    """Extract comprehensive features from media for analysis"""
    
    @staticmethod
    def extract_image_features(image):
        """Extract all image features"""
        features = {}
        
        # Convert to numpy array if PIL Image
        if isinstance(image, Image.Image):
            img_array = np.array(image)
        else:
            img_array = image
        
        # Basic statistics
        features.update(FeatureExtractor._extract_basic_stats(img_array))
        
        # Texture features
        features.update(FeatureExtractor._extract_texture_features(img_array))
        
        # Frequency features
        features.update(FeatureExtractor._extract_frequency_features(img_array))
        
        # Color features
        features.update(FeatureExtractor._extract_color_features(img_array))
        
        # Noise features
        features.update(FeatureExtractor._extract_noise_features(img_array))
        
        # Compression artifacts
        features.update(FeatureExtractor._detect_compression_artifacts(img_array))
        
        return features
    
    @staticmethod
    def extract_audio_features(audio, sr):
        """Extract all audio features"""
        features = {}
        
        # Time domain features
        features.update(FeatureExtractor._extract_time_domain_features(audio))
        
        # Frequency domain features
        features.update(FeatureExtractor._extract_frequency_domain_features(audio, sr))
        
        # Spectral features
        features.update(FeatureExtractor._extract_spectral_features(audio, sr))
        
        # Voice features
        features.update(FeatureExtractor._extract_voice_features(audio, sr))
        
        # Noise features
        features.update(FeatureExtractor._extract_audio_noise_features(audio))
        
        return features
    
    @staticmethod
    def extract_video_features(frames):
        """Extract video features from frame sequence"""
        features = {}
        
        # Spatial features per frame
        spatial_features = []
        for frame in frames:
            spatial_features.append(
                FeatureExtractor.extract_image_features(frame)
            )
        
        # Temporal features
        features.update(FeatureExtractor._extract_temporal_features(frames))
        
        # Motion features
        features.update(FeatureExtractor._extract_motion_features(frames))
        
        # Aggregate spatial features
        features['avg_spatial_features'] = {
            key: np.mean([f[key] for f in spatial_features])
            for key in spatial_features[0].keys()
        }
        
        return features
    
    @staticmethod
    def _extract_basic_stats(img_array):
        """Extract basic statistical features"""
        features = {}
        
        features['mean'] = float(np.mean(img_array))
        features['std'] = float(np.std(img_array))
        features['variance'] = float(np.var(img_array))
        features['min'] = float(np.min(img_array))
        features['max'] = float(np.max(img_array))
        features['range'] = float(np.max(img_array) - np.min(img_array))
        features['skewness'] = float(FeatureExtractor._skewness(img_array))
        features['kurtosis'] = float(FeatureExtractor._kurtosis(img_array))
        
        return features
    
    @staticmethod
    def _extract_texture_features(img_array):
        """Extract texture features using GLCM"""
        features = {}
        
        # Convert to grayscale if needed
        if len(img_array.shape) == 3:
            gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        else:
            gray = img_array
        
        # Calculate gradient magnitude
        grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        grad_mag = np.sqrt(grad_x**2 + grad_y**2)
        
        features['gradient_mean'] = float(np.mean(grad_mag))
        features['gradient_std'] = float(np.std(grad_mag))
        features['gradient_energy'] = float(np.sum(grad_mag**2))
        
        # Local Binary Pattern (simplified)
        lbp = FeatureExtractor._local_binary_pattern(gray)
        features['lbp_histogram'] = lbp.tolist()
        
        return features
    
    @staticmethod
    def _extract_frequency_features(img_array):
        """Extract frequency domain features"""
        features = {}
        
        # Convert to grayscale
        if len(img_array.shape) == 3:
            gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        else:
            gray = img_array
        
        # Apply FFT
        f = fft2(gray)
        fshift = fftshift(f)
        magnitude = np.log(np.abs(fshift) + 1)
        
        # Frequency statistics
        features['freq_mean'] = float(np.mean(magnitude))
        features['freq_std'] = float(np.std(magnitude))
        
        # Energy in different frequency bands
        h, w = magnitude.shape
        center_h, center_w = h//2, w//2
        
        # Low frequencies (center)
        low = magnitude[center_h-30:center_h+30, center_w-30:center_w+30]
        features['low_freq_energy'] = float(np.sum(low))
        
        # High frequencies (edges)
        high = np.concatenate([
            magnitude[:50, :].flatten(),
            magnitude[-50:, :].flatten(),
            magnitude[:, :50].flatten(),
            magnitude[:, -50:].flatten()
        ])
        features['high_freq_energy'] = float(np.sum(high))
        
        if features['high_freq_energy'] > 0:
            features['freq_ratio'] = features['low_freq_energy'] / features['high_freq_energy']
        else:
            features['freq_ratio'] = 0
        
        return features
    
    @staticmethod
    def _extract_color_features(img_array):
        """Extract color-based features"""
        features = {}
        
        if len(img_array.shape) == 3:
            # Color channels
            r, g, b = img_array[:,:,0], img_array[:,:,1], img_array[:,:,2]
            
            # Channel statistics
            features['r_mean'] = float(np.mean(r))
            features['g_mean'] = float(np.mean(g))
            features['b_mean'] = float(np.mean(b))
            
            # Channel correlations
            features['corr_rg'] = float(np.corrcoef(r.flatten(), g.flatten())[0,1])
            features['corr_rb'] = float(np.corrcoef(r.flatten(), b.flatten())[0,1])
            features['corr_gb'] = float(np.corrcoef(g.flatten(), b.flatten())[0,1])
            
            # Color moments
            features['color_moment_1'] = float(np.mean(r))  # Simple version
            features['color_moment_2'] = float(np.std(g))
            features['color_moment_3'] = float(FeatureExtractor._skewness(b))
        
        return features
    
    @staticmethod
    def _extract_noise_features(img_array):
        """Extract noise-related features"""
        features = {}
        
        # Convert to grayscale
        if len(img_array.shape) == 3:
            gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        else:
            gray = img_array
        
        # Estimate noise using median filter
        denoised = cv2.medianBlur(gray.astype(np.uint8), 5)
        noise = cv2.absdiff(gray.astype(np.uint8), denoised)
        
        features['noise_mean'] = float(np.mean(noise))
        features['noise_std'] = float(np.std(noise))
        features['snr'] = float(np.mean(gray) / (np.mean(noise) + 1e-8))
        
        return features
    
    @staticmethod
    def _detect_compression_artifacts(img_array):
        """Detect JPEG compression artifacts"""
        features = {}
        
        # Convert to grayscale
        if len(img_array.shape) == 3:
            gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        else:
            gray = img_array
        
        # Check for blocking artifacts
        h, w = gray.shape
        block_size = 8
        
        block_differences = []
        for i in range(0, h - block_size, block_size):
            for j in range(0, w - block_size, block_size):
                block = gray[i:i+block_size, j:j+block_size]
                
                # Check boundaries
                if i + block_size < h:
                    vertical_diff = np.abs(block[-1, :] - gray[i+block_size, j:j+block_size])
                    block_differences.append(np.mean(vertical_diff))
                
                if j + block_size < w:
                    horizontal_diff = np.abs(block[:, -1] - gray[i:i+block_size, j+block_size])
                    block_differences.append(np.mean(horizontal_diff))
        
        if block_differences:
            features['blocking_artifact'] = float(np.mean(block_differences))
            features['blocking_std'] = float(np.std(block_differences))
        else:
            features['blocking_artifact'] = 0
            features['blocking_std'] = 0
        
        return features
    
    @staticmethod
    def _extract_time_domain_features(audio):
        """Extract time domain audio features"""
        features = {}
        
        features['audio_mean'] = float(np.mean(audio))
        features['audio_std'] = float(np.std(audio))
        features['audio_rms'] = float(np.sqrt(np.mean(audio**2)))
        features['audio_peak'] = float(np.max(np.abs(audio)))
        features['audio_crest_factor'] = features['audio_peak'] / (features['audio_rms'] + 1e-8)
        
        return features
    
    @staticmethod
    def _extract_frequency_domain_features(audio, sr):
        """Extract frequency domain audio features"""
        features = {}
        
        # Compute spectrogram
        f, t, Sxx = scipy.signal.spectrogram(audio, sr)
        
        features['spec_mean'] = float(np.mean(Sxx))
        features['spec_std'] = float(np.std(Sxx))
        features['spec_max'] = float(np.max(Sxx))
        
        # Dominant frequencies
        dominant_freqs = f[np.argmax(Sxx, axis=0)]
        features['dominant_freq_mean'] = float(np.mean(dominant_freqs))
        features['dominant_freq_std'] = float(np.std(dominant_freqs))
        
        return features
    
    @staticmethod
    def _extract_spectral_features(audio, sr):
        """Extract spectral audio features"""
        features = {}
        
        # Spectral centroid
        spectral_centroids = librosa.feature.spectral_centroid(y=audio, sr=sr)[0]
        features['spectral_centroid_mean'] = float(np.mean(spectral_centroids))
        features['spectral_centroid_std'] = float(np.std(spectral_centroids))
        
        # Spectral bandwidth
        spectral_bandwidth = librosa.feature.spectral_bandwidth(y=audio, sr=sr)[0]
        features['spectral_bandwidth_mean'] = float(np.mean(spectral_bandwidth))
        
        # Spectral rolloff
        spectral_rolloff = librosa.feature.spectral_rolloff(y=audio, sr=sr)[0]
        features['spectral_rolloff_mean'] = float(np.mean(spectral_rolloff))
        
        # Spectral contrast
        spectral_contrast = librosa.feature.spectral_contrast(y=audio, sr=sr)
        features['spectral_contrast_mean'] = float(np.mean(spectral_contrast))
        
        return features
    
    @staticmethod
    def _extract_voice_features(audio, sr):
        """Extract voice-specific features"""
        features = {}
        
        # Pitch
        pitches, magnitudes = librosa.piptrack(y=audio, sr=sr)
        pitch_values = pitches[pitches > 0]
        
        if len(pitch_values) > 0:
            features['pitch_mean'] = float(np.mean(pitch_values))
            features['pitch_std'] = float(np.std(pitch_values))
            features['pitch_range'] = float(np.max(pitch_values) - np.min(pitch_values))
        else:
            features['pitch_mean'] = 0
            features['pitch_std'] = 0
            features['pitch_range'] = 0
        
        # Formants (simplified)
        try:
            lpc_coeffs = librosa.lpc(audio, order=12)
            roots = np.roots(lpc_coeffs)
            roots = [r for r in roots if np.imag(r) >= 0]
            angles = np.arctan2(np.imag(roots), np.real(roots))
            formants = sorted(angles * (sr / (2 * np.pi)))
            
            if len(formants) >= 3:
                features['f1'] = float(formants[0])
                features['f2'] = float(formants[1])
                features['f3'] = float(formants[2])
            else:
                features['f1'] = 0
                features['f2'] = 0
                features['f3'] = 0
        except:
            features['f1'] = 0
            features['f2'] = 0
            features['f3'] = 0
        
        return features
    
    @staticmethod
    def _extract_audio_noise_features(audio):
        """Extract audio noise features"""
        features = {}
        
        # Signal-to-noise ratio estimation
        # Simple method: assume first 10% is silence for noise floor
        noise_length = int(len(audio) * 0.1)
        if noise_length > 0:
            noise_floor = np.mean(np.abs(audio[:noise_length])**2)
            signal_power = np.mean(np.abs(audio)**2)
            
            if noise_floor > 0:
                features['snr'] = float(10 * np.log10(signal_power / noise_floor))
            else:
                features['snr'] = float('inf')
        else:
            features['snr'] = 0
        
        # Harmonic-to-noise ratio
        try:
            hnr = librosa.effects.harmonic(audio)
            features['hnr'] = float(np.mean(hnr))
        except:
            features['hnr'] = 0
        
        return features
    
    @staticmethod
    def _extract_temporal_features(frames):
        """Extract temporal features from video frames"""
        features = {}
        
        if len(frames) < 2:
            return features
        
        # Frame-to-frame differences
        diffs = []
        for i in range(1, len(frames)):
            frame1 = frames[i-1]
            frame2 = frames[i]
            
            # Convert to grayscale
            if len(frame1.shape) == 3:
                gray1 = cv2.cvtColor(frame1, cv2.COLOR_RGB2GRAY)
                gray2 = cv2.cvtColor(frame2, cv2.COLOR_RGB2GRAY)
            else:
                gray1 = frame1
                gray2 = frame2
            
            diff = np.mean(np.abs(gray1.astype(float) - gray2.astype(float)))
            diffs.append(diff)
        
        features['temporal_diff_mean'] = float(np.mean(diffs))
        features['temporal_diff_std'] = float(np.std(diffs))
        features['temporal_diff_max'] = float(np.max(diffs))
        
        # Temporal consistency
        if len(diffs) > 1:
            features['temporal_consistency'] = float(1 - np.std(diffs) / (np.mean(diffs) + 1e-8))
        else:
            features['temporal_consistency'] = 1.0
        
        return features
    
    @staticmethod
    def _extract_motion_features(frames):
        """Extract motion features using optical flow"""
        features = {}
        
        if len(frames) < 2:
            return features
        
        motion_magnitudes = []
        motion_angles = []
        
        for i in range(1, min(len(frames), 30)):  # Limit to 30 frames for performance
            frame1 = frames[i-1]
            frame2 = frames[i]
            
            # Convert to grayscale
            if len(frame1.shape) == 3:
                gray1 = cv2.cvtColor(frame1, cv2.COLOR_RGB2GRAY)
                gray2 = cv2.cvtColor(frame2, cv2.COLOR_RGB2GRAY)
            else:
                gray1 = frame1
                gray2 = frame2
            
            # Calculate optical flow
            flow = cv2.calcOpticalFlowFarneback(
                gray1, gray2, None,
                pyr_scale=0.5, levels=3, winsize=15,
                iterations=3, poly_n=5, poly_sigma=1.2,
                flags=0
            )
            
            # Calculate magnitude and angle
            mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
            
            motion_magnitudes.append(np.mean(mag))
            motion_angles.append(np.mean(ang))
        
        if motion_magnitudes:
            features['motion_magnitude_mean'] = float(np.mean(motion_magnitudes))
            features['motion_magnitude_std'] = float(np.std(motion_magnitudes))
            features['motion_angle_mean'] = float(np.mean(motion_angles))
            features['motion_angle_std'] = float(np.std(motion_angles))
            
            # Motion consistency
            features['motion_consistency'] = float(
                1 - np.std(motion_magnitudes) / (np.mean(motion_magnitudes) + 1e-8)
            )
        
        return features
    
    @staticmethod
    def _local_binary_pattern(gray_img, P=8, R=1):
        """Compute Local Binary Pattern histogram"""
        from skimage.feature import local_binary_pattern
        
        try:
            lbp = local_binary_pattern(gray_img, P, R, method='uniform')
            n_bins = int(lbp.max() + 1)
            hist, _ = np.histogram(lbp.ravel(), bins=n_bins, range=(0, n_bins))
            hist = hist.astype(float) / (hist.sum() + 1e-8)
            return hist
        except:
            # Return default histogram if skimage not available
            return np.zeros(10)
    
    @staticmethod
    def _skewness(data):
        """Calculate skewness"""
        if len(data.shape) > 1:
            data = data.flatten()
        
        n = len(data)
        if n < 3:
            return 0
        
        mean = np.mean(data)
        std = np.std(data)
        
        if std == 0:
            return 0
        
        return np.sum((data - mean)**3) / (n * std**3)
    
    @staticmethod
    def _kurtosis(data):
        """Calculate kurtosis"""
        if len(data.shape) > 1:
            data = data.flatten()
        
        n = len(data)
        if n < 4:
            return 0
        
        mean = np.mean(data)
        std = np.std(data)
        
        if std == 0:
            return 0
        
        return np.sum((data - mean)**4) / (n * std**4) - 3
    
    @staticmethod
    def compute_image_hash(image):
        """Compute perceptual hash of image"""
        if isinstance(image, Image.Image):
            img_array = np.array(image)
        else:
            img_array = image
        
        # Convert to grayscale
        if len(img_array.shape) == 3:
            gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        else:
            gray = img_array
        
        # Resize to 8x8
        small = cv2.resize(gray, (8, 8))
        
        # Compute hash
        hash_str = hashlib.md5(small.tobytes()).hexdigest()
        
        return hash_str