import os
import json
import hashlib
import datetime
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from werkzeug.utils import secure_filename
import numpy as np
from PIL import Image
import io
import base64
import warnings
warnings.filterwarnings('ignore')

# Import custom modules
from models.image_detector import DeepfakeImageDetector
from models.audio_detector import DeepfakeAudioDetector
from models.video_detector import DeepfakeVideoDetector
from models.ensemble_model import EnsembleDetector
from utils.preprocessor import DataPreprocessor
from utils.feature_extractor import FeatureExtractor
from utils.logger import ForensicLogger

# Initialize Flask app
app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key-here-change-in-production'
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024  # 500MB max file size
app.config['ALLOWED_EXTENSIONS'] = {
    'image': {'jpg', 'jpeg', 'png', 'gif', 'bmp', 'webp'},
    'audio': {'mp3', 'wav', 'ogg', 'm4a', 'flac'},
    'video': {'mp4', 'avi', 'mov', 'mkv', 'webm'}
}

# Initialize CORS
CORS(app, resources={r"/*": {"origins": "*"}})

# Initialize detectors (lazy loading)
image_detector = None
audio_detector = None
video_detector = None
ensemble_detector = None

# Initialize logger
logger = ForensicLogger()

def allowed_file(file_type, filename):
    """Check if file extension is allowed"""
    ext = filename.rsplit('.', 1)[1].lower() if '.' in filename else ''
    return ext in app.config['ALLOWED_EXTENSIONS'].get(file_type, set())

def load_models():
    """Load AI models (lazy loading)"""
    global image_detector, audio_detector, video_detector, ensemble_detector
    
    if image_detector is None:
        print("Loading Image Detection Model...")
        image_detector = DeepfakeImageDetector()
    
    if audio_detector is None:
        print("Loading Audio Detection Model...")
        audio_detector = DeepfakeAudioDetector()
    
    if video_detector is None:
        print("Loading Video Detection Model...")
        video_detector = DeepfakeVideoDetector()
    
    if ensemble_detector is None:
        print("Loading Ensemble Model...")
        ensemble_detector = EnsembleDetector()

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.datetime.now().isoformat(),
        'message': 'DeepShield API is running!'
    })

@app.route('/api/analyze/image', methods=['POST'])
def analyze_image():
    """Analyze image for deepfakes"""
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
        
        file = request.files['file']
        
        if file.filename == '':
            return jsonify({'error': 'Empty filename'}), 400
        
        if not allowed_file('image', file.filename):
            return jsonify({'error': 'File type not allowed. Allowed: jpg, jpeg, png, gif, bmp, webp'}), 400
        
        # Load models
        load_models()
        
        # Read and preprocess image
        image_bytes = file.read()
        image = Image.open(io.BytesIO(image_bytes))
        
        # Generate file hash for tracking
        file_hash = hashlib.sha256(image_bytes).hexdigest()[:16]
        
        # Analyze image
        result = image_detector.analyze(image)
        
        # Extract additional features
        features = FeatureExtractor.extract_image_features(image)
        
        # Get ensemble prediction
        ensemble_result = ensemble_detector.predict_image(image) if ensemble_detector else None
        
        # Log the analysis
        logger.log_analysis('image', file_hash, result, ensemble_result, file.filename, len(image_bytes))
        
        # Generate heatmap
        heatmap = image_detector.generate_heatmap(image)
        
        # Convert heatmap to base64 for sending
        heatmap_buffer = io.BytesIO()
        heatmap.save(heatmap_buffer, format='PNG')
        heatmap_base64 = base64.b64encode(heatmap_buffer.getvalue()).decode()
        
        return jsonify({
            'success': True,
            'result': result['label'],
            'confidence': float(result['confidence']),
            'probabilities': result.get('probabilities', {}),
            'ensemble_confidence': float(ensemble_result['confidence']) if ensemble_result else None,
            'features': features,
            'heatmap': f"data:image/png;base64,{heatmap_base64}",
            'analysis_time': result.get('analysis_time', 0),
            'file_hash': file_hash,
            'recommendations': generate_recommendations(result)
        })
        
    except Exception as e:
        print(f"Error in image analysis: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/analyze/audio', methods=['POST'])
def analyze_audio():
    """Analyze audio for deepfakes"""
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
        
        file = request.files['file']
        
        if file.filename == '':
            return jsonify({'error': 'Empty filename'}), 400
        
        if not allowed_file('audio', file.filename):
            return jsonify({'error': 'File type not allowed. Allowed: mp3, wav, ogg, m4a, flac'}), 400
        
        # Load models
        load_models()
        
        # Save temporarily
        temp_path = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(file.filename))
        file.save(temp_path)
        
        # Analyze audio
        result = audio_detector.analyze(temp_path)
        
        # Generate file hash
        with open(temp_path, 'rb') as f:
            file_hash = hashlib.sha256(f.read()).hexdigest()[:16]
        
        # Generate spectrogram
        spectrogram = audio_detector.generate_spectrogram(temp_path)
        
        # Clean up
        os.remove(temp_path)
        
        return jsonify({
            'success': True,
            'result': result['label'],
            'confidence': float(result['confidence']),
            'voice_characteristics': result.get('voice_features', {}),
            'spectrogram': spectrogram,
            'analysis_time': result.get('analysis_time', 0),
            'file_hash': file_hash
        })
        
    except Exception as e:
        print(f"Error in audio analysis: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/analyze/video', methods=['POST'])
def analyze_video():
    """Analyze video for deepfakes"""
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
        
        file = request.files['file']
        
        if file.filename == '':
            return jsonify({'error': 'Empty filename'}), 400
        
        if not allowed_file('video', file.filename):
            return jsonify({'error': 'File type not allowed. Allowed: mp4, avi, mov, mkv, webm'}), 400
        
        # Load models
        load_models()
        
        # Save temporarily
        temp_path = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(file.filename))
        file.save(temp_path)
        
        # Analyze video frame by frame
        result = video_detector.analyze(temp_path)
        
        # Generate file hash
        with open(temp_path, 'rb') as f:
            file_hash = hashlib.sha256(f.read()).hexdigest()[:16]
        
        # Generate frame analysis
        frame_analysis = video_detector.analyze_frames(temp_path)
        
        # Clean up
        os.remove(temp_path)
        
        return jsonify({
            'success': True,
            'result': result['label'],
            'confidence': float(result['confidence']),
            'frame_count': result.get('frame_count', 0),
            'temporal_analysis': result.get('temporal_features', {}),
            'frame_analysis': frame_analysis,
            'analysis_time': result.get('analysis_time', 0),
            'file_hash': file_hash
        })
        
    except Exception as e:
        print(f"Error in video analysis: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/reports', methods=['GET'])
def get_reports():
    """Get recent analysis reports"""
    try:
        limit = request.args.get('limit', 50, type=int)
        reports = logger.get_recent_reports(limit)
        return jsonify({
            'success': True,
            'reports': reports
        })
    except Exception as e:
        print(f"Error getting reports: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/report/<file_hash>', methods=['GET'])
def get_report(file_hash):
    """Get specific analysis report by file hash"""
    try:
        report = logger.get_report(file_hash)
        if report:
            return jsonify({
                'success': True,
                'report': report
            })
        else:
            return jsonify({'error': 'Report not found'}), 404
    except Exception as e:
        print(f"Error getting report: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/stats', methods=['GET'])
def get_stats():
    """Get system statistics"""
    try:
        stats = logger.get_statistics()
        return jsonify({
            'success': True,
            'statistics': stats
        })
    except Exception as e:
        print(f"Error getting stats: {str(e)}")
        return jsonify({'error': str(e)}), 500

def generate_recommendations(result):
    """Generate recommendations based on analysis"""
    recommendations = []
    
    confidence = result.get('confidence', 0)
    
    if confidence > 0.8:
        recommendations.append("High confidence detection - This media shows strong signs of manipulation")
        recommendations.append("Review the highlighted areas in the heatmap for specific artifacts")
        recommendations.append("Consider cross-referencing with original source if available")
    elif confidence > 0.5:
        recommendations.append("Moderate confidence - Some indicators of manipulation detected")
        recommendations.append("Consider additional verification methods")
        recommendations.append("Review metadata and compression patterns")
    else:
        recommendations.append("Low confidence - Media appears authentic")
        recommendations.append("No significant manipulation detected")
        recommendations.append("Standard verification checks passed")
    
    return recommendations

# Create upload folder if it doesn't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

if __name__ == '__main__':
    print("=" * 60)
    print("🔍 DeepShield Forensic Engine")
    print("=" * 60)
    print("🚀 Server starting...")
    print(f"📁 Upload folder: {app.config['UPLOAD_FOLDER']}")
    print(f"📍 API URL: http://127.0.0.1:5000")
    print("\n📡 Available endpoints:")
    print("  ✓ GET  /api/health")
    print("  ✓ POST /api/analyze/image")
    print("  ✓ POST /api/analyze/audio")
    print("  ✓ POST /api/analyze/video")
    print("  ✓ GET  /api/reports")
    print("  ✓ GET  /api/stats")
    print("=" * 60)
    print("⚡ Press CTRL+C to stop the server")
    print("=" * 60)
    
    # Run without SocketIO to avoid the error
    app.run(debug=True, host='127.0.0.1', port=5000, threaded=True)