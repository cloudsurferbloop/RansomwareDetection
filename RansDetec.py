#!/usr/bin/env python3
"""
Ransomware Detection System with ML
Combines rule-based detection with Isolation Forest
"""

import os
import time
import math
import json
import logging
import pickle
import numpy as np
from datetime import datetime, timedelta
from collections import defaultdict, deque
from flask import Flask, request, jsonify, send_file
from flask_socketio import SocketIO, emit
from werkzeug.utils import secure_filename

# Try importing sklearn, fall back if not available
try:
    from sklearn.ensemble import IsolationForest
    from sklearn.preprocessing import StandardScaler
    SKLEARN_AVAILABLE = True
except Exception:
    IsolationForest = None
    StandardScaler = None
    SKLEARN_AVAILABLE = False
    print('scikit-learn not available; using fallback ML detector')
import threading

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.config['SECRET_KEY'] = os.getenv('FLASK_SECRET_KEY', 'dev-key-change-in-production')
app.config['UPLOAD_FOLDER'] = os.getenv('UPLOAD_DIR', 'uploads')
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max
app.config['MODEL_DIR'] = 'models'

socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading')

# Make sure directories exist
for folder in [app.config['UPLOAD_FOLDER'], app.config['MODEL_DIR']]:
    if not os.path.exists(folder):
        os.makedirs(folder)
        logger.info(f"Created directory: {folder}")


class FeatureExtractor:
    """Extracts features from file activity for ML detection"""
    
    def __init__(self, window_seconds=10):
        self.window_seconds = window_seconds
        self.activity_window = deque(maxlen=1000)
        # Common file types targeted by ransomware
        self.high_value_extensions = {
            '.docx', '.pdf', '.xlsx', '.jpg', '.jpeg', '.png', 
            '.db', '.sql', '.txt', '.pptx', '.zip', '.rar'
        }
        
    def add_activity(self, activity_type, filepath, entropy=0.0):
        self.activity_window.append({
            'type': activity_type,
            'path': filepath,
            'entropy': entropy,
            'timestamp': time.time(),
            'is_high_value': self._is_high_value_file(filepath)
        })
    
    def _is_high_value_file(self, filepath):
        ext = os.path.splitext(filepath)[1].lower()
        return ext in self.high_value_extensions
    
    def extract_features(self):
        """Extract 10 features from recent activity"""
        current_time = time.time()
        cutoff_time = current_time - self.window_seconds
        
        recent_activity = [a for a in self.activity_window if a['timestamp'] >= cutoff_time]
        
        if not recent_activity:
            return np.zeros(10)
        
        files_modified = sum(1 for a in recent_activity if a['type'] == 'modified')
        files_renamed = sum(1 for a in recent_activity if a['type'] == 'renamed')
        files_deleted = sum(1 for a in recent_activity if a['type'] == 'deleted')
        
        # Calculate average entropy
        modified_entropies = [a['entropy'] for a in recent_activity 
                             if a['type'] == 'modified' and a['entropy'] > 0]
        avg_entropy = np.mean(modified_entropies) if modified_entropies else 0.0
        
        high_value_count = sum(1 for a in recent_activity if a['is_high_value'])
        high_value_ratio = high_value_count / len(recent_activity) if recent_activity else 0.0
        
        modification_rate = files_modified / self.window_seconds
        rename_rate = files_renamed / self.window_seconds
        
        # How many different directories are affected
        unique_dirs = len(set(os.path.dirname(a['path']) for a in recent_activity))
        
        high_entropy_count = sum(1 for e in modified_entropies if e > 7.5)
        
        # Detect activity bursts
        activity_burst = 1.0 if len(recent_activity) > 20 else 0.0
        
        feature_vector = np.array([
            files_modified,
            files_renamed,
            files_deleted,
            avg_entropy,
            high_value_ratio,
            modification_rate,
            rename_rate,
            unique_dirs,
            high_entropy_count,
            activity_burst
        ])
        
        return feature_vector


class MLAnomalyDetector:
    """ML-based anomaly detection using Isolation Forest"""
    
    def __init__(self, contamination=0.1):
        self.is_trained = False
        self.training_data = []
        self.feature_names = [
            'files_modified', 'files_renamed', 'files_deleted',
            'avg_entropy', 'high_value_ratio', 'modification_rate',
            'rename_rate', 'directory_spread', 'high_entropy_count',
            'activity_burst'
        ]

        if SKLEARN_AVAILABLE:
            self.model = IsolationForest(
                contamination=contamination,
                random_state=42,
                n_estimators=100,
                max_samples='auto',
                bootstrap=False
            )
            self.scaler = StandardScaler()
        else:
            # Fallback when sklearn isn't available
            self.model = None
            self.scaler = None
            self.is_trained = True  # so predict() works with fallback
        
    def add_training_sample(self, feature_vector):
        """Add sample during training phase"""
        if SKLEARN_AVAILABLE:
            self.training_data.append(feature_vector)
            logger.debug(f"Training sample collected. Total: {len(self.training_data)}")
    
    def train(self, min_samples=50):
        """Train model on collected samples"""
        if not SKLEARN_AVAILABLE:
            logger.info("SKLearn not available — skipping ML training (using fallback heuristic)")
            self.is_trained = True
            return True

        if len(self.training_data) < min_samples:
            logger.warning(f"Not enough training data. Need {min_samples}, have {len(self.training_data)}")
            return False

        X = np.array(self.training_data)
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled)
        self.is_trained = True

        logger.info(f"Model trained on {len(self.training_data)} samples")
        return True
    
    def predict(self, feature_vector):
        """
        Predict if behavior is anomalous
        Returns (anomaly_score, is_anomaly)
        """
        if not self.is_trained:
            logger.warning("Model not trained yet. Returning neutral prediction.")
            return 0.0, False

        # Use sklearn model if available
        if SKLEARN_AVAILABLE and self.model is not None and self.scaler is not None:
            X = feature_vector.reshape(1, -1)
            X_scaled = self.scaler.transform(X)

            anomaly_score = self.model.score_samples(X_scaled)[0]
            prediction = self.model.predict(X_scaled)[0]
            is_anomaly = (prediction == -1)

            # normalize score to 0-1
            normalized_score = max(0, -anomaly_score) / 2.0

            return normalized_score, is_anomaly

        # Simple heuristic fallback
        try:
            fv = np.asarray(feature_vector, dtype=float).ravel()
        except Exception:
            fv = np.array(feature_vector, dtype=float)

        # Basic scoring based on key features
        avg_entropy = float(fv[3]) if len(fv) > 3 else 0.0
        files_modified = float(fv[0]) if len(fv) > 0 else 0.0
        modification_rate = float(fv[5]) if len(fv) > 5 else 0.0
        activity_burst = float(fv[9]) if len(fv) > 9 else 0.0

        score = 0.0
        score += min(1.0, avg_entropy / 8.0) * 0.6
        score += min(1.0, files_modified / 20.0) * 0.2
        score += min(1.0, modification_rate / 5.0) * 0.1
        score += activity_burst * 0.1

        score = max(0.0, min(1.0, score))
        is_anomaly = score >= 0.6
        return float(score), bool(is_anomaly)
    
    def save_model(self, filepath):
        if not self.is_trained:
            logger.warning("Cannot save untrained model")
            return False

        model_data = {
            'model': self.model if SKLEARN_AVAILABLE else None,
            'scaler': self.scaler if SKLEARN_AVAILABLE else None,
            'training_samples': len(self.training_data),
            'feature_names': self.feature_names,
            'fallback': not SKLEARN_AVAILABLE
        }

        try:
            with open(filepath, 'wb') as f:
                pickle.dump(model_data, f)
            logger.info(f"Model saved to {filepath}")
            return True
        except Exception as e:
            logger.error(f"Failed to save model: {e}")
            return False
    
    def load_model(self, filepath):
        if not os.path.exists(filepath):
            logger.warning(f"Model file not found: {filepath}")
            return False
        try:
            with open(filepath, 'rb') as f:
                model_data = pickle.load(f)

            if SKLEARN_AVAILABLE and model_data.get('model') is not None:
                self.model = model_data['model']
                self.scaler = model_data.get('scaler', self.scaler)
                self.feature_names = model_data.get('feature_names', self.feature_names)
                self.is_trained = True
            else:
                # fallback mode
                self.model = None
                self.scaler = None
                self.feature_names = model_data.get('feature_names', self.feature_names)
                self.is_trained = True

            logger.info(f"Model loaded from {filepath} (trained on {model_data.get('training_samples', 0)} samples)")
            return True
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            return False


class EnhancedRansomwareDetector:
    """Main detector combining rules and ML"""
    
    def __init__(self):
        # Known ransomware extensions
        self.suspicious_extensions = [
            '.encrypted', '.locked', '.crypto', '.crypt', 
            '.crypted', '.enc', '.ransomware', '.cerber',
            '.locky', '.zepto', '.thor', '.aaa', '.abc'
        ]
        
        # Keywords commonly found in ransom notes
        self.ransom_note_keywords = [
            'ransom', 'bitcoin', 'decrypt', 'payment', 'btc', 
            'restore', 'recover', 'unlock', 'crypto', 'wallet',
            'encryption', 'encrypted', 'pay', 'deadline'
        ]
        
        self.feature_extractor = FeatureExtractor(window_seconds=10)
        self.ml_detector = MLAnomalyDetector(contamination=0.1)
        
        # Risk scoring weights - TODO: tune these based on real data
        self.weights = {
            'extension': 30,
            'keyword': 25,
            'entropy': 30,
            'ml_anomaly': 40,
            'rapid_modification': 20,
            'rename_burst': 25
        }
        
        self.risk_threshold = 70
        self.training_mode = False
        
        self.reset_stats()
        
    def enable_training_mode(self):
        self.training_mode = True
        logger.info("Training mode ENABLED - collecting normal behavior samples")
    
    def disable_training_mode(self):
        self.training_mode = False
        success = self.ml_detector.train()
        if success:
            logger.info("Training mode DISABLED - model trained successfully")
            model_path = os.path.join(app.config['MODEL_DIR'], 'ransomware_model.pkl')
            self.ml_detector.save_model(model_path)
        else:
            logger.warning("Training mode DISABLED - insufficient data to train model")
    
    def calc_file_entropy(self, filepath):
        """Calculate Shannon entropy of file - encrypted files have high entropy"""
        try:
            with open(filepath, 'rb') as f:
                data = f.read(8192)  # read first 8KB
            
            if len(data) == 0:
                return 0.0, False
            
            freq = defaultdict(int)
            for byte in data:
                freq[byte] += 1
            
            entropy = 0.0
            data_len = len(data)
            for count in freq.values():
                probability = count / data_len
                entropy -= probability * math.log2(probability)
            
            is_high_entropy = entropy > 7.5
            return round(entropy, 2), is_high_entropy
            
        except Exception as e:
            logger.warning(f"Error calculating entropy for {filepath}: {e}")
            return 0.0, False
    
    def analyze_file(self, filepath):
        """Main analysis function"""
        flags = []
        entropy_value = 0.0
        risk_score = 0
        
        if not os.path.exists(filepath) or os.path.isdir(filepath):
            logger.error(f"Invalid file path: {filepath}")
            return None
        
        filename = os.path.basename(filepath).lower()
        
        # Clean up display name (remove timestamp prefix if present)
        display_name = filename
        parts = filename.split('_', 2)
        if len(parts) >= 3 and parts[0].isdigit() and parts[1].isdigit():
            display_name = parts[2]
        
        file_size = os.path.getsize(filepath)
        
        # Check for suspicious extensions
        extension_flagged = False
        for ext in self.suspicious_extensions:
            if filepath.lower().endswith(ext):
                flags.append({'type': 'ext', 'text': f'Extension: {ext}'})
                extension_flagged = True
                risk_score += self.weights['extension']
                self.stats['extensionFlags'] += 1
                logger.warning(f"Suspicious extension detected: {ext} in {display_name}")
                break
        
        # Check for ransom keywords
        keyword_flagged = False
        for keyword in self.ransom_note_keywords:
            if keyword in display_name:
                flags.append({'type': 'kw', 'text': f'Keyword: {keyword}'})
                keyword_flagged = True
                risk_score += self.weights['keyword']
                self.stats['keywordFlags'] += 1
                logger.warning(f"Ransom keyword '{keyword}' found in {display_name}")
                break
        
        # Entropy check
        entropy_flagged = False
        try:
            entropy_value, high_entropy = self.calc_file_entropy(filepath)
            if high_entropy:
                flags.append({'type': 'ent', 'text': f'High entropy: {entropy_value}/8.0'})
                entropy_flagged = True
                risk_score += self.weights['entropy']
                self.stats['entropyFlags'] += 1
                logger.warning(f"High entropy ({entropy_value}) detected in {display_name}")
        except Exception as e:
            logger.error(f"Entropy check failed for {display_name}: {e}")
        
        # ML detection
        self.feature_extractor.add_activity('modified', filepath, entropy_value)
        feature_vector = self.feature_extractor.extract_features()
        
        ml_anomaly_detected = False
        ml_score = 0.0
        
        if self.training_mode:
            # Collect clean samples during training
            if risk_score == 0:
                self.ml_detector.add_training_sample(feature_vector)
                logger.debug("Added sample to training data (normal behavior)")
        else:
            # Use ML model for prediction
            ml_score, ml_anomaly_detected = self.ml_detector.predict(feature_vector)
            
            if ml_anomaly_detected:
                flags.append({
                    'type': 'ml', 
                    'text': f'ML Anomaly: {ml_score:.2f} confidence'
                })
                risk_score += self.weights['ml_anomaly']
                self.stats['mlFlags'] += 1
                logger.warning(f"ML anomaly detected for {display_name} (score: {ml_score:.2f})")
        
        # Determine threat level
        if risk_score >= self.risk_threshold:
            threat_level = 'danger'
            self.stats['danger'] += 1
            logger.error(f"HIGH THREAT DETECTED: {display_name} - Risk Score: {risk_score}")
        elif risk_score >= self.risk_threshold * 0.5:
            threat_level = 'warning'
            self.stats['suspicious'] += 1
            logger.warning(f"Suspicious activity: {display_name} - Risk Score: {risk_score}")
        else:
            threat_level = 'safe'
            self.stats['safe'] += 1
        
        self.stats['total'] += 1
        self.stats['totalFlags'] += len(flags)
        
        result = {
            'name': display_name,
            'size': file_size,
            'ent': entropy_value,
            'flags': flags,
            'level': threat_level,
            'time': datetime.now().isoformat(),
            'hasExt': extension_flagged,
            'hasKw': keyword_flagged,
            'hasEnt': entropy_flagged,
            'hasMl': ml_anomaly_detected,
            'riskScore': risk_score,
            'mlScore': round(ml_score, 3),
            'trainingMode': self.training_mode
        }
        
        logger.info(f"Scan complete: {display_name} - {threat_level} (Risk: {risk_score})")
        return result
    
    def get_stats(self):
        stats = self.stats.copy()
        stats['mlTrained'] = self.ml_detector.is_trained
        stats['trainingMode'] = self.training_mode
        stats['trainingSamples'] = len(self.ml_detector.training_data)
        return stats
    
    def reset_stats(self):
        self.stats = {
            'total': 0,
            'safe': 0,
            'suspicious': 0,
            'danger': 0,
            'extensionFlags': 0,
            'keywordFlags': 0,
            'entropyFlags': 0,
            'mlFlags': 0,
            'totalFlags': 0
        }
        logger.info("Statistics reset")


# Initialize detector
detector = EnhancedRansomwareDetector()

# Load existing model if available
model_path = os.path.join(app.config['MODEL_DIR'], 'ransomware_model.pkl')
if os.path.exists(model_path):
    detector.ml_detector.load_model(model_path)
    logger.info("Loaded existing ML model")


@app.route('/')
def index():
    """Serve main page"""
    html_path = 'index.html'
    if os.path.exists(html_path):
        return send_file(html_path)
    else:
        return """
        <html>
        <body style="font-family: Arial; padding: 50px; background: #f8f9fa;">
            <h1 style="color: #dc2626;">⚠️ HTML File Not Found</h1>
            <p>Please make sure <code>index.html</code> is in the same folder as this Python script.</p>
        </body>
        </html>
        """, 404


@app.route('/api/scan', methods=['POST'])
def scan_file():
    """Handle file upload and scan"""
    if 'files' not in request.files:
        return jsonify({'error': 'No files provided'}), 400
    
    files = request.files.getlist('files')
    results = []
    
    logger.info(f"Starting scan of {len(files)} file(s)")
    
    for file in files:
        if file.filename == '':
            continue
        
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        
        try:
            file.save(filepath)
            result = detector.analyze_file(filepath)
            
            if result:
                results.append(result)
                socketio.emit('file_scanned', result)
            
            # Clean up
            os.remove(filepath)
            
        except Exception as e:
            logger.error(f"Error processing {filename}: {e}")
            if os.path.exists(filepath):
                try:
                    os.remove(filepath)
                except:
                    pass
    
    return jsonify({
        'results': results,
        'stats': detector.get_stats()
    })


@app.route('/api/stats', methods=['GET'])
def get_stats():
    return jsonify(detector.get_stats())


@app.route('/api/reset', methods=['POST'])
def reset_stats():
    detector.reset_stats()
    socketio.emit('stats_reset', {})
    return jsonify({'success': True})


@app.route('/api/training/enable', methods=['POST'])
def enable_training():
    detector.enable_training_mode()
    return jsonify({'success': True, 'trainingMode': True})


@app.route('/api/training/disable', methods=['POST'])
def disable_training():
    detector.disable_training_mode()
    return jsonify({
        'success': True, 
        'trainingMode': False,
        'modelTrained': detector.ml_detector.is_trained
    })


@app.route('/api/training/status', methods=['GET'])
def training_status():
    return jsonify({
        'trainingMode': detector.training_mode,
        'modelTrained': detector.ml_detector.is_trained,
        'trainingSamples': len(detector.ml_detector.training_data)
    })


@socketio.on('connect')
def handle_connect():
    logger.info('Client connected via WebSocket')
    emit('connection_response', {'status': 'connected'})


@socketio.on('disconnect')
def handle_disconnect():
    logger.info('Client disconnected')

if __name__ == '__main__':
    print("="*80)
    print("🛡️  ENHANCED RANSOMWARE DETECTION SYSTEM (ML-Enabled)")
    print("="*80)
    print(f"Upload directory: {app.config['UPLOAD_FOLDER']}")
    print(f"Model directory: {app.config['MODEL_DIR']}")
    print(f"ML Model status: {'TRAINED ✓' if detector.ml_detector.is_trained else 'NOT TRAINED - Use training mode first'}")
    print("\nStarting Flask server on http://localhost:5000")
    print("Press Ctrl+C to stop")
    print("="*80 + "\n")
    
    socketio.run(app, debug=True, host='0.0.0.0', port=5000)
