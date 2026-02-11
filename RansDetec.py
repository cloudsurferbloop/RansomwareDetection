#!/usr/bin/env python3

import os
import time
import math
import json
import logging
from datetime import datetime
from collections import defaultdict
from flask import Flask, request, jsonify, send_file
from flask_socketio import SocketIO, emit
from werkzeug.utils import secure_filename
import threading

# Set up basic logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.config['SECRET_KEY'] = os.getenv('FLASK_SECRET_KEY', 'dev-key-change-in-production')
app.config['UPLOAD_FOLDER'] = os.getenv('UPLOAD_DIR', 'uploads')
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

socketio = SocketIO(app, cors_allowed_origins="*")

# make sure upload folder exists
if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])
    logger.info(f"Created upload directory: {app.config['UPLOAD_FOLDER']}")


class RansomwareDetector:
    # Core detection logic for identifying potential ransomware
    
    def __init__(self):
        # Extensions commonly used by ransomware
        self.suspicious_extensions = ['.encrypted', '.locked', '.crypto', '.crypt', 
                                       '.crypted', '.enc', '.ransomware']
        
        # Common keywords found in ransom notes
        self.ransom_note_keywords = ['ransom', 'bitcoin', 'decrypt', 'payment', 
                                      'btc', 'restore', 'recover', 'unlock']
        
        self.reset_stats()
        
    def calc_file_entropy(self, filepath):
        """Calculate Shannon entropy - high entropy suggests encryption"""
        try:
            with open(filepath, 'rb') as f:
                # just read first chunk, don't need whole file
                data = f.read(8192)
            
            if len(data) == 0:
                return 0.0, False
            
            # count byte frequencies
            freq = defaultdict(int)
            for byte in data:
                freq[byte] += 1
            
            entropy = 0.0
            data_len = len(data)
            for count in freq.values():
                probability = count / data_len
                entropy -= probability * math.log2(probability)
            
            # Encrypted files typically have entropy > 7.5
            is_high_entropy = entropy > 7.5
            return round(entropy, 2), is_high_entropy
            
        except Exception as e:
            logger.warning(f"Error calculating entropy for {filepath}: {e}")
            return 0.0, False
    
    def analyze_file(self, filepath):
        # Main analysis function
        flags = []
        entropy_value = 0.0
        
        if not os.path.exists(filepath) or os.path.isdir(filepath):
            logger.error(f"Invalid file path: {filepath}")
            return None
        
        filename = os.path.basename(filepath).lower()
        file_size = os.path.getsize(filepath)
        
        # Check 1: Look for suspicious file extensions
        extension_flagged = False
        for ext in self.suspicious_extensions:
            if filepath.lower().endswith(ext):
                flags.append({'type': 'ext', 'text': f'Extension: {ext}'})
                extension_flagged = True
                self.stats['extensionFlags'] += 1
                logger.warning(f"Suspicious extension detected: {ext} in {filename}")
                break
        
        # Check 2: Scan filename for ransom-related keywords
        keyword_flagged = False
        for keyword in self.ransom_note_keywords:
            if keyword in filename:
                flags.append({'type': 'kw', 'text': f'Keyword: {keyword}'})
                keyword_flagged = True
                self.stats['keywordFlags'] += 1
                logger.warning(f"Ransom keyword '{keyword}' found in {filename}")
                break
        
        # Check 3: Analyze file entropy
        entropy_flagged = False
        try:
            entropy_value, high_entropy = self.calc_file_entropy(filepath)
            if high_entropy:
                flags.append({'type': 'ent', 'text': f'High entropy: {entropy_value}/8.0'})
                entropy_flagged = True
                self.stats['entropyFlags'] += 1
                logger.warning(f"High entropy ({entropy_value}) detected in {filename}")
        except Exception as e:
            logger.error(f"Entropy check failed for {filename}: {e}")
        
        # Determine overall threat level based on flags
        flag_count = len(flags)
        if flag_count == 0:
            threat_level = 'safe'
            self.stats['safe'] += 1
        elif flag_count == 1:
            threat_level = 'warning'
            self.stats['suspicious'] += 1
        else:  # 2 or more flags
            threat_level = 'danger'
            self.stats['danger'] += 1
            logger.error(f"THREAT DETECTED: {filename} has {flag_count} red flags!")
        
        self.stats['total'] += 1
        self.stats['totalFlags'] += flag_count
        
        result = {
            'name': os.path.basename(filepath),
            'size': file_size,
            'ent': entropy_value,
            'flags': flags,
            'level': threat_level,
            'time': datetime.now().isoformat(),
            'hasExt': extension_flagged,
            'hasKw': keyword_flagged,
            'hasEnt': entropy_flagged
        }
        
        logger.info(f"Scan complete: {filename} - {threat_level}")
        return result
    
    def get_stats(self):
        return self.stats
    
    def reset_stats(self):
        """Reset all counters back to zero"""
        self.stats = {
            'total': 0,
            'safe': 0,
            'suspicious': 0,
            'danger': 0,
            'extensionFlags': 0,
            'keywordFlags': 0,
            'entropyFlags': 0,
            'totalFlags': 0
        }
        logger.info("Statistics reset")


# Initialize detector
detector = RansomwareDetector()


@app.route('/')
def index():
    """Main page"""
    html_path = 'index.html'
    if os.path.exists(html_path):
        return send_file(html_path)
    else:
        logger.error("index.html not found!")
        return """
        <html>
        <body style="font-family: Arial; padding: 50px; background: #f8f9fa;">
            <h1 style="color: #dc2626;">⚠️ HTML File Not Found</h1>
            <p>Please make sure <code>index.html</code> is in the same folder as this Python script.</p>
            <p>Your folder should look like this:</p>
            <pre style="background: #fff; padding: 20px; border-radius: 8px;">
ransomware-detector/
├── ransomwaredetector.py  (this file)
└── index.html             (the dashboard HTML)
            </pre>
            <p>Once you add the HTML file, refresh this page.</p>
        </body>
        </html>
        """, 404


@app.route('/api/scan', methods=['POST'])
def scan_file():
    # Handle file upload and scanning
    if 'files' not in request.files:
        logger.warning("Scan request received with no files")
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
            logger.info(f"Analyzing: {filename}")
            
            result = detector.analyze_file(filepath)
            
            if result:
                results.append(result)
                # Send real-time update to connected clients
                socketio.emit('file_scanned', result)
            
            # cleanup - remove temp file
            os.remove(filepath)
            
        except Exception as e:
            logger.error(f"Error processing {filename}: {e}")
            # still try to clean up
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
    # Return current statistics
    return jsonify(detector.get_stats())


@app.route('/api/reset', methods=['POST'])
def reset_stats():
    """Clear all stats and notify clients"""
    logger.info("Resetting all statistics")
    detector.reset_stats()
    socketio.emit('stats_reset', {})
    return jsonify({'success': True})


@socketio.on('connect')
def handle_connect():
    logger.info('Client connected via WebSocket')
    emit('connection_response', {'status': 'connected'})


@socketio.on('disconnect')
def handle_disconnect():
    logger.info('Client disconnected')


if __name__ == '__main__':
    print("="*80)
    print("🛡️  RANSOMWARE DETECTION WEB SERVER")
    print("="*80)
    print(f"Upload directory: {app.config['UPLOAD_FOLDER']}")
    print(f"Secret key: {'[from environment]' if os.getenv('FLASK_SECRET_KEY') else '[using dev default]'}")
    print("\nMake sure 'index.html' is in the same folder as this script!")
    print("\nStarting Flask server on http://localhost:5000")
    print("Press Ctrl+C to stop")
    print("="*80 + "\n")
    
    socketio.run(app, debug=True, host='0.0.0.0', port=5000)