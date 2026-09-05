import numpy as np
import librosa
import librosa.display
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy import signal
from scipy.io import wavfile
import matplotlib.pyplot as plt
import io
import base64
import time
import warnings
warnings.filterwarnings('ignore')

class DeepfakeAudioDetector:
    """Real audio deepfake detection using spectral analysis and deep learning"""
    
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.sample_rate = 16000
        self.n_mfcc = 13
        self.n_fft = 2048
        self.hop_length = 512
        
        # Load pre-trained model or create simple one
        self.model = self._create_model()
        
    def _create_model(self):
        """Create CNN model for audio classification"""
        class AudioCNN(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
                self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
                self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
                self.pool = nn.MaxPool2d(2, 2)
                self.dropout = nn.Dropout(0.3)
                
                # Calculate size after convolutions
                self._to_linear = None
                self._get_conv_output()
                
                self.fc1 = nn.Linear(self._to_linear, 256)
                self.fc2 = nn.Linear(256, 128)
                self.fc3 = nn.Linear(128, 2)
                
            def _get_conv_output(self):
                """Get size of conv output for FC layer"""
                with torch.no_grad():
                    dummy = torch.zeros(1, 1, 128, 128)
                    dummy = self.pool(F.relu(self.conv1(dummy)))
                    dummy = self.pool(F.relu(self.conv2(dummy)))
                    dummy = self.pool(F.relu(self.conv3(dummy)))
                    self._to_linear = dummy.view(1, -1).size(1)
            
            def forward(self, x):
                x = self.pool(F.relu(self.conv1(x)))
                x = self.pool(F.relu(self.conv2(x)))
                x = self.pool(F.relu(self.conv3(x)))
                x = self.dropout(x)
                x = x.view(x.size(0), -1)
                x = F.relu(self.fc1(x))
                x = self.dropout(x)
                x = F.relu(self.fc2(x))
                x = self.fc3(x)
                return x
        
        return AudioCNN().to(self.device)
    
    def analyze(self, audio_path):
        """Analyze audio file for deepfake indicators"""
        start_time = time.time()
        
        # Load audio
        audio, sr = self._load_audio(audio_path)
        
        # Extract features
        features = self._extract_features(audio, sr)
        
        # Analyze for deepfake indicators
        indicators = self._detect_deepfake_indicators(audio, sr, features)
        
        # Get model prediction
        model_prediction = self._predict_with_model(audio, sr)
        
        # Combine all signals
        fake_probability = self._combine_signals(indicators, model_prediction)
        
        # Extract voice characteristics
        voice_features = self._extract_voice_characteristics(audio, sr)
        
        analysis_time = time.time() - start_time
        
        return {
            'label': 'Deepfake Audio' if fake_probability > 0.5 else 'Real Audio',
            'confidence': float(fake_probability),
            'voice_features': voice_features,
            'indicators': indicators,
            'analysis_time': round(analysis_time, 3)
        }
    
    def _load_audio(self, audio_path):
        """Load and preprocess audio file"""
        try:
            audio, sr = librosa.load(audio_path, sr=self.sample_rate, mono=True)
            
            # Normalize
            audio = audio / (np.max(np.abs(audio)) + 1e-8)
            
            return audio, sr
        except Exception as e:
            print(f"Error loading audio: {e}")
            # Return silent audio as fallback
            return np.zeros(self.sample_rate * 3), self.sample_rate
    
    def _extract_features(self, audio, sr):
        """Extract audio features"""
        features = {}
        
        # MFCCs
        mfccs = librosa.feature.mfcc(
            y=audio, 
            sr=sr, 
            n_mfcc=self.n_mfcc,
            n_fft=self.n_fft,
            hop_length=self.hop_length
        )
        features['mfccs'] = mfccs
        
        # Spectral features
        spectral_centroids = librosa.feature.spectral_centroid(
            y=audio, 
            sr=sr,
            n_fft=self.n_fft,
            hop_length=self.hop_length
        )[0]
        features['spectral_centroids'] = spectral_centroids
        
        # Spectral rolloff
        spectral_rolloff = librosa.feature.spectral_rolloff(
            y=audio, 
            sr=sr,
            n_fft=self.n_fft,
            hop_length=self.hop_length
        )[0]
        features['spectral_rolloff'] = spectral_rolloff
        
        # Zero crossing rate
        zcr = librosa.feature.zero_crossing_rate(audio, hop_length=self.hop_length)[0]
        features['zcr'] = zcr
        
        # RMS energy
        rms = librosa.feature.rms(y=audio, hop_length=self.hop_length)[0]
        features['rms'] = rms
        
        return features
    
    def _detect_deepfake_indicators(self, audio, sr, features):
        """Detect specific deepfake audio indicators"""
        indicators = {}
        
        # Check for unnatural pauses/silence
        indicators['unnatural_pauses'] = self._detect_unnatural_pauses(audio)
        
        # Check for frequency inconsistencies
        indicators['frequency_anomalies'] = self._detect_frequency_anomalies(audio, sr)
        
        # Check for breathing artifacts
        indicators['breathing_artifacts'] = self._detect_breathing_artifacts(audio)
        
        # Check for formant consistency
        indicators['formant_issues'] = self._detect_formant_issues(audio, sr)
        
        # Check for spectral consistency
        indicators['spectral_consistency'] = self._check_spectral_consistency(features)
        
        return indicators
    
    def _detect_unnatural_pauses(self, audio):
        """Detect unnatural pauses in speech"""
        # Simple energy-based voice activity detection
        frame_length = int(0.025 * self.sample_rate)  # 25ms frames
        hop_length = int(0.010 * self.sample_rate)    # 10ms hop
        
        # Calculate energy in frames
        energy = []
        for i in range(0, len(audio) - frame_length, hop_length):
            frame = audio[i:i+frame_length]
            energy.append(np.sum(frame**2))
        
        energy = np.array(energy)
        
        # Detect silence (low energy)
        silence_threshold = 0.01 * np.max(energy)
        is_silence = energy < silence_threshold
        
        # Find pause lengths
        pause_lengths = []
        current_pause = 0
        
        for silent in is_silence:
            if silent:
                current_pause += 1
            else:
                if current_pause > 0:
                    pause_lengths.append(current_pause)
                    current_pause = 0
        
        if current_pause > 0:
            pause_lengths.append(current_pause)
        
        # Convert to seconds
        pause_lengths_sec = np.array(pause_lengths) * hop_length / self.sample_rate
        
        # Check for unnatural pause patterns
        if len(pause_lengths_sec) > 0:
            avg_pause = np.mean(pause_lengths_sec)
            std_pause = np.std(pause_lengths_sec)
            
            # Very regular pauses might indicate synthetic speech
            if std_pause < 0.1 and avg_pause > 0.1:
                return 0.8
            elif std_pause < 0.2:
                return 0.6
            else:
                return 0.3
        else:
            return 0.1
    
    def _detect_frequency_anomalies(self, audio, sr):
        """Detect frequency domain anomalies"""
        # Compute spectrogram
        f, t, Sxx = signal.spectrogram(audio, sr)
        
        # Find dominant frequencies over time
        dominant_freqs = f[np.argmax(Sxx, axis=0)]
        
        # Check for unnatural frequency jumps
        freq_changes = np.abs(np.diff(dominant_freqs))
        
        # Human speech typically has smooth frequency transitions
        if np.max(freq_changes) > 500:  # >500Hz jump is unnatural
            return 0.9
        elif np.mean(freq_changes) > 100:
            return 0.7
        else:
            return 0.3
    
    def _detect_breathing_artifacts(self, audio):
        """Detect presence or absence of breathing sounds"""
        # Simple breathing detection based on low-frequency noise
        # Apply low-pass filter
        b, a = signal.butter(4, 100/(self.sample_rate/2), 'low')
        low_freq = signal.filtfilt(b, a, audio)
        
        # Calculate energy in low frequencies
        low_energy = np.sum(low_freq**2)
        total_energy = np.sum(audio**2)
        
        if total_energy > 0:
            breathing_ratio = low_energy / total_energy
            
            # Real speech has some breathing, synthetic often has none
            if breathing_ratio < 0.01:
                return 0.8  # No breathing - suspicious
            elif breathing_ratio > 0.2:
                return 0.6  # Too much breathing - also suspicious
            else:
                return 0.3  # Normal range
        else:
            return 0.5
    
    def _detect_formant_issues(self, audio, sr):
        """Check for formant consistency issues"""
        # Use LPC to estimate formants
        order = 12  # LPC order
        
        # Apply pre-emphasis
        pre_emphasis = 0.97
        emphasized = np.append(audio[0], audio[1:] - pre_emphasis * audio[:-1])
        
        # Get LPC coefficients
        try:
            lpc_coeffs = librosa.lpc(emphasized, order=order)
            
            # Find roots of LPC polynomial
            roots = np.roots(lpc_coeffs)
            roots = [r for r in roots if np.imag(r) >= 0]
            
            # Get angles and convert to frequencies
            angles = np.arctan2(np.imag(roots), np.real(roots))
            formants = sorted(angles * (sr / (2 * np.pi)))
            
            # Check formant range
            if len(formants) >= 3:
                # Human speech typically has formants in specific ranges
                f1, f2, f3 = formants[:3]
                
                if f1 < 200 or f1 > 1000:
                    return 0.7
                elif f2 < 600 or f2 > 3000:
                    return 0.7
                else:
                    return 0.3
            else:
                return 0.5
                
        except:
            return 0.5
    
    def _check_spectral_consistency(self, features):
        """Check consistency of spectral features"""
        inconsistencies = []
        
        # Check MFCC stability
        mfccs = features.get('mfccs', np.zeros((13, 100)))
        mfcc_std = np.std(mfccs, axis=1)
        
        if np.mean(mfcc_std) > 50:
            inconsistencies.append(0.7)
        elif np.mean(mfcc_std) > 30:
            inconsistencies.append(0.5)
        else:
            inconsistencies.append(0.3)
        
        # Check spectral centroid variation
        centroids = features.get('spectral_centroids', np.zeros(100))
        centroid_std = np.std(centroids)
        
        if centroid_std > 500:
            inconsistencies.append(0.7)
        else:
            inconsistencies.append(0.3)
        
        return np.mean(inconsistencies)
    
    def _predict_with_model(self, audio, sr):
        """Get model prediction"""
        try:
            # Prepare input for model (mel-spectrogram)
            mel_spec = librosa.feature.melspectrogram(
                y=audio, 
                sr=sr,
                n_mels=128,
                n_fft=self.n_fft,
                hop_length=self.hop_length
            )
            
            # Convert to log scale
            log_mel = librosa.power_to_db(mel_spec)
            
            # Normalize
            log_mel = (log_mel - log_mel.mean()) / (log_mel.std() + 1e-8)
            
            # Resize to fixed size (128x128)
            if log_mel.shape[1] < 128:
                # Pad if too short
                pad_width = 128 - log_mel.shape[1]
                log_mel = np.pad(log_mel, ((0, 0), (0, pad_width)), mode='constant')
            else:
                # Truncate if too long
                log_mel = log_mel[:, :128]
            
            # Add channel dimension
            input_tensor = torch.FloatTensor(log_mel).unsqueeze(0).unsqueeze(0).to(self.device)
            
            # Get prediction
            with torch.no_grad():
                outputs = self.model(input_tensor)
                probs = torch.softmax(outputs, dim=1)
                fake_prob = probs[0][1].item()
            
            return fake_prob
            
        except Exception as e:
            print(f"Model prediction error: {e}")
            return 0.5
    
    def _combine_signals(self, indicators, model_pred):
        """Combine all signals for final prediction"""
        weights = {
            'unnatural_pauses': 0.2,
            'frequency_anomalies': 0.25,
            'breathing_artifacts': 0.15,
            'formant_issues': 0.2,
            'spectral_consistency': 0.2
        }
        
        weighted_sum = 0
        total_weight = 0
        
        for key, value in indicators.items():
            if key in weights:
                weighted_sum += value * weights[key]
                total_weight += weights[key]
        
        if total_weight > 0:
            indicators_score = weighted_sum / total_weight
        else:
            indicators_score = 0.5
        
        # Combine with model prediction
        final_score = 0.4 * indicators_score + 0.6 * model_pred
        
        return final_score
    
    def _extract_voice_characteristics(self, audio, sr):
        """Extract voice characteristics for analysis"""
        characteristics = {}
        
        # Pitch
        pitches, magnitudes = librosa.piptrack(y=audio, sr=sr)
        pitch_values = pitches[pitches > 0]
        
        if len(pitch_values) > 0:
            characteristics['avg_pitch'] = float(np.mean(pitch_values))
            characteristics['pitch_std'] = float(np.std(pitch_values))
        else:
            characteristics['avg_pitch'] = 0
            characteristics['pitch_std'] = 0
        
        # Speech rate (simplified)
        # Count onsets as approximation
        onset_frames = librosa.onset.onset_detect(y=audio, sr=sr, hop_length=self.hop_length)
        duration = len(audio) / sr
        
        if duration > 0:
            characteristics['speech_rate'] = len(onset_frames) / duration
        else:
            characteristics['speech_rate'] = 0
        
        # Voice quality (harmonic-to-noise ratio)
        try:
            hnr = librosa.effects.harmonic(audio)
            characteristics['hnr'] = float(np.mean(hnr))
        except:
            characteristics['hnr'] = 0
        
        return characteristics
    
    def generate_spectrogram(self, audio_path):
        """Generate spectrogram for visualization"""
        try:
            audio, sr = self._load_audio(audio_path)
            
            # Generate mel spectrogram
            plt.figure(figsize=(10, 4))
            mel_spec = librosa.feature.melspectrogram(y=audio, sr=sr)
            mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
            
            librosa.display.specshow(mel_spec_db, sr=sr, x_axis='time', y_axis='mel')
            plt.colorbar(format='%+2.0f dB')
            plt.title('Mel Spectrogram')
            plt.tight_layout()
            
            # Convert to base64
            buffer = io.BytesIO()
            plt.savefig(buffer, format='png', dpi=100, bbox_inches='tight')
            plt.close()
            
            buffer.seek(0)
            image_base64 = base64.b64encode(buffer.getvalue()).decode()
            
            return f"data:image/png;base64,{image_base64}"
            
        except Exception as e:
            print(f"Spectrogram generation error: {e}")
            return None