import numpy as np
from collections import defaultdict
import json

class EnsembleDetector:
    """Ensemble model combining multiple detection approaches"""
    
    def __init__(self):
        self.weights = {
            'image': {
                'cnn': 0.4,
                'frequency': 0.2,
                'statistical': 0.2,
                'artifact': 0.2
            },
            'audio': {
                'spectral': 0.35,
                'temporal': 0.25,
                'voice': 0.25,
                'noise': 0.15
            },
            'video': {
                'spatial': 0.3,
                'temporal': 0.3,
                'motion': 0.2,
                'face': 0.2
            }
        }
        
        self.confidence_thresholds = {
            'high': 0.8,
            'medium': 0.6,
            'low': 0.4
        }
        
        # Cache for results
        self.cache = {}
        
    def predict_image(self, image):
        """Ensemble prediction for image"""
        # This would normally call multiple models
        # For now, simulate ensemble prediction
        
        # Mock predictions from different models
        predictions = {
            'cnn': np.random.uniform(0.3, 0.9),
            'frequency': np.random.uniform(0.3, 0.9),
            'statistical': np.random.uniform(0.3, 0.9),
            'artifact': np.random.uniform(0.3, 0.9)
        }
        
        # Calculate weighted score
        weighted_score = sum(
            predictions[model] * self.weights['image'][model]
            for model in predictions
        )
        
        # Calculate confidence based on agreement
        agreement = 1 - np.std(list(predictions.values()))
        confidence = (weighted_score + agreement) / 2
        
        return {
            'score': float(weighted_score),
            'confidence': float(confidence),
            'individual_predictions': predictions,
            'agreement': float(agreement)
        }
    
    def predict_audio(self, audio_features):
        """Ensemble prediction for audio"""
        predictions = {
            'spectral': np.random.uniform(0.3, 0.9),
            'temporal': np.random.uniform(0.3, 0.9),
            'voice': np.random.uniform(0.3, 0.9),
            'noise': np.random.uniform(0.3, 0.9)
        }
        
        weighted_score = sum(
            predictions[model] * self.weights['audio'][model]
            for model in predictions
        )
        
        agreement = 1 - np.std(list(predictions.values()))
        confidence = (weighted_score + agreement) / 2
        
        return {
            'score': float(weighted_score),
            'confidence': float(confidence),
            'individual_predictions': predictions,
            'agreement': float(agreement)
        }
    
    def predict_video(self, video_features):
        """Ensemble prediction for video"""
        predictions = {
            'spatial': np.random.uniform(0.3, 0.9),
            'temporal': np.random.uniform(0.3, 0.9),
            'motion': np.random.uniform(0.3, 0.9),
            'face': np.random.uniform(0.3, 0.9)
        }
        
        weighted_score = sum(
            predictions[model] * self.weights['video'][model]
            for model in predictions
        )
        
        agreement = 1 - np.std(list(predictions.values()))
        confidence = (weighted_score + agreement) / 2
        
        return {
            'score': float(weighted_score),
            'confidence': float(confidence),
            'individual_predictions': predictions,
            'agreement': float(agreement)
        }
    
    def get_confidence_level(self, confidence):
        """Get confidence level category"""
        if confidence >= self.confidence_thresholds['high']:
            return 'HIGH'
        elif confidence >= self.confidence_thresholds['medium']:
            return 'MEDIUM'
        elif confidence >= self.confidence_thresholds['low']:
            return 'LOW'
        else:
            return 'VERY_LOW'
    
    def cache_result(self, file_hash, result):
        """Cache analysis result"""
        self.cache[file_hash] = {
            'result': result,
            'timestamp': time.time()
        }
    
    def get_cached_result(self, file_hash):
        """Get cached result if exists and not expired"""
        if file_hash in self.cache:
            cached = self.cache[file_hash]
            # Cache for 1 hour
            if time.time() - cached['timestamp'] < 3600:
                return cached['result']
        return None