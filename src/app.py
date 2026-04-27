import os
import sys
import json
import threading
import cv2
import numpy as np
from flask import Flask, render_template, request, jsonify, redirect, url_for

# Ensure src/ is on path
sys.path.insert(0, os.path.dirname(__file__))

from preprocess import detect_and_crop_face, decode_base64_image
from embedding_store import get_embedding, get_store
from register import add_new_face

app = Flask(__name__, template_folder='../templates')

FACES_DIR = os.path.join(os.path.dirname(__file__), '..', 'faces')
UPLOAD_DIR = os.path.join(os.path.dirname(__file__), '..', 'uploads')
ERP_DATA_PATH = os.path.join(os.path.dirname(__file__), '..', 'erp_data.json')

os.makedirs(FACES_DIR, exist_ok=True)
os.makedirs(UPLOAD_DIR, exist_ok=True)

_lock = threading.Lock()
_training_status = {'busy': False, 'last': 'idle'}


def load_erp_data():
    if os.path.exists(ERP_DATA_PATH):
        with open(ERP_DATA_PATH, 'r') as f:
            return json.load(f)
    return {}


def save_erp_data(data):
    with open(ERP_DATA_PATH, 'w') as f:
        json.dump(data, f, indent=2)


# ─── Routes ────────────────────────────────────────────────────────────────────

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/add_face', methods=['GET'])
def add_face_page():
    return render_template('add_face.html')


@app.route('/api/add_face', methods=['POST'])
def api_add_face():
    """
    Expects multipart form: roll_number, name, branch, year, image (file)
    """
    roll = request.form.get('roll_number', '').strip().upper()
    name = request.form.get('name', '').strip()
    branch = request.form.get('branch', '').strip()
    year = request.form.get('year', '').strip()
    image_file = request.files.get('image')

    if not roll or not image_file:
        return jsonify({'error': 'Roll number and image are required.'}), 400

    # Save uploaded image
    ext = image_file.filename.rsplit('.', 1)[-1] if '.' in image_file.filename else 'jpg'
    save_path = os.path.join(UPLOAD_DIR, f'{roll}.{ext}')
    image_file.save(save_path)

    # Verify face is detectable
    face = detect_and_crop_face(save_path)
    if face is None:
        return jsonify({'error': 'No face detected in the uploaded image. Please use a clear frontal photo.'}), 400

    # Save ERP profile
    erp_data = load_erp_data()
    erp_data[roll] = {
        'roll_number': roll,
        'name': name,
        'branch': branch,
        'year': year,
        'photo': f'/uploads/{roll}.{ext}'
    }
    save_erp_data(erp_data)

    # Fine-tune in background — reload model when done
    def register_async():
        _training_status['busy'] = True
        _training_status['last'] = f'Registering {roll}...'
        try:
            add_new_face(roll, save_path)          # ~1 s, no training
            _training_status['last'] = f'{roll} registered and ready.'
        except Exception as e:
            _training_status['last'] = f'Registration error: {e}'
        finally:
            _training_status['busy'] = False

    t = threading.Thread(target=register_async, daemon=True)
    t.start()

    return jsonify({'success': True, 'message': f'Face for {roll} registered. Processing embeddings...'})


@app.route('/api/recognize', methods=['POST'])
def api_recognize():
    data = request.get_json()
    if not data or 'image' not in data:
        return jsonify({'error': 'No image provided'}), 400

    img_bgr = decode_base64_image(data['image'])
    if img_bgr is None:
        return jsonify({'error': 'Could not decode image'}), 400

    face = detect_and_crop_face(img_bgr)
    if face is None:
        return jsonify({'face_detected': False})

    embedding = get_embedding(face)            # no lock needed, stateless
    roll, confidence = get_store().search(embedding, threshold=0.40)

    if roll is None:
        return jsonify({'face_detected': True, 'unknown': True, 'confidence': round(confidence, 3)})

    erp_data = load_erp_data()
    student  = erp_data.get(roll, {'roll_number': roll, 'name': 'Unknown'})
    return jsonify({
        'face_detected': True,
        'unknown':       False,
        'roll_number':   roll,
        'name':          student.get('name', ''),
        'confidence':    round(confidence, 3)
    })


@app.route('/erp/<roll_number>')
def erp_profile(roll_number):
    erp_data = load_erp_data()
    student = erp_data.get(roll_number.upper())
    if not student:
        return render_template('erp.html', student=None, roll=roll_number.upper())
    return render_template('erp.html', student=student)


@app.route('/api/status')
def api_status():
    return jsonify(_training_status)


@app.route('/uploads/<filename>')
def uploaded_file(filename):
    from flask import send_from_directory
    return send_from_directory(UPLOAD_DIR, filename)


@app.route('/favicon.ico')
def favicon():
    return '', 204


if __name__ == '__main__':
    import ssl
    cert_path = os.path.join(os.path.dirname(__file__), '..', 'cert.pem')
    key_path = os.path.join(os.path.dirname(__file__), '..', 'key.pem')

    if os.path.exists(cert_path) and os.path.exists(key_path):
        app.run(debug=True, host='0.0.0.0', port=5000, ssl_context=(cert_path, key_path))
    else:
        print("⚠  No SSL cert found — running HTTP. Camera will only work on localhost.")
        print("   Run: openssl req -x509 -newkey rsa:2048 -keyout key.pem -out cert.pem -days 365 -nodes -subj '/CN=localhost'")
        app.run(debug=True, host='0.0.0.0', port=5000)
