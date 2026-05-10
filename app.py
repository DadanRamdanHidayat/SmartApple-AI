# =========================
# IMPORT LIBRARY
# =========================
from flask import Flask, render_template, request, jsonify, redirect, url_for
import os
from datetime import datetime
from werkzeug.utils import secure_filename
import random

# =========================
# MODE MODEL (ON / OFF)
# =========================
USE_MODEL = True  # 👉 ubah True kalau model sudah ada

if USE_MODEL:
    import numpy as np
    from tensorflow.keras.models import load_model
    from tensorflow.keras.preprocessing import image as keras_image

# =========================
# INISIALISASI FLASK
# =========================
app = Flask(__name__)

app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 5 * 1024 * 1024

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs('static/img/uploads', exist_ok=True)

# =========================
# LOAD MODEL
# =========================
if USE_MODEL:
    model = load_model('model/model.h5')
    IMG_SIZE = (224, 224)

# =========================
# LABEL KELAS (4 KELAS)
# =========================
CLASS_LABELS = ['Blotch_Apple', 'Normal_Apple', 'Rot_Apple', 'Scab_Apple']

# Optional (biar user friendly)
LABEL_TRANSLATE = {
    'Blotch_Apple': 'Bercak (Blotch)',
    'Normal_Apple': 'Apel Sehat',
    'Rot_Apple': 'Busuk',
    'Scab_Apple': 'Scab'
}

# =========================
# INFO PER KELAS
# =========================
CLASS_INFO = {
    'Normal_Apple': {
        'kondisi': 'Sehat',
        'estimasi_panen': 'Siap dikonsumsi',
        'solusi': 'Tidak perlu tindakan khusus.'
    },
    'Blotch_Apple': {
        'kondisi': 'Terdapat bercak',
        'estimasi_panen': 'Perlu seleksi',
        'solusi': 'Gunakan fungisida dan pisahkan buah.'
    },
    'Rot_Apple': {
        'kondisi': 'Busuk',
        'estimasi_panen': 'Tidak layak',
        'solusi': 'Buang buah agar tidak menyebar.'
    },
    'Scab_Apple': {
        'kondisi': 'Terinfeksi scab',
        'estimasi_panen': 'Perlu penanganan',
        'solusi': 'Perawatan lanjutan dengan fungisida.'
    }
}

# =========================
# HISTORY
# =========================
history_store = []
history_id_counter = 1

# =========================
# ROUTES
# =========================
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/deteksi')
def deteksi():
    return render_template('detectionpage.html')

@app.route('/tentang')
def tentang():
    return render_template('aboutpage.html')

# =========================
# RIWAYAT + STATS
# =========================
@app.route('/riwayat')
def riwayat():

    stats = {
        'total': len(history_store),
        'normal': sum(1 for h in history_store if h['label'] == 'Normal_Apple'),
        'blotch': sum(1 for h in history_store if h['label'] == 'Blotch_Apple'),
        'rot': sum(1 for h in history_store if h['label'] == 'Rot_Apple'),
        'scab': sum(1 for h in history_store if h['label'] == 'Scab_Apple'),
    }

    return render_template(
        'historypage.html',
        histories=history_store[::-1],
        stats=stats
    )

# =========================
# HAPUS DATA
# =========================
@app.route('/riwayat/hapus/<int:item_id>')
def hapus_riwayat(item_id):
    global history_store
    history_store = [h for h in history_store if h['id'] != item_id]
    return redirect(url_for('riwayat'))

@app.route('/riwayat/hapus-semua')
def hapus_semua():
    global history_store
    history_store = []
    return redirect(url_for('riwayat'))

# =========================
# VALIDASI FILE
# =========================
ALLOWED_EXT = {'jpg', 'jpeg', 'png', 'webp'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXT

# =========================
# PREPROCESS
# =========================
def preprocess_image(img_path):
    img = keras_image.load_img(img_path, target_size=IMG_SIZE)
    img_array = keras_image.img_to_array(img)
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# =========================
# PREDICT
# =========================
@app.route('/predict', methods=['POST'])
def predict():
    global history_id_counter

    if 'file' not in request.files:
        return jsonify({'error': 'Tidak ada file'}), 400

    file = request.files['file']

    if file.filename == '' or not allowed_file(file.filename):
        return jsonify({'error': 'Format file tidak valid'}), 400

    filename = secure_filename(file.filename)
    filename = datetime.now().strftime('%Y%m%d_%H%M%S_') + filename

    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)

    import shutil
    shutil.copy(filepath, os.path.join('static/img/uploads', filename))

    try:
        # =========================
        # MODE AI
        # =========================
        if USE_MODEL:
            img = preprocess_image(filepath)

            prediction = model.predict(img)[0]
            class_index = int(np.argmax(prediction))

            label_raw = CLASS_LABELS[class_index]
            label = LABEL_TRANSLATE.get(label_raw, label_raw)

            confidence = float(prediction[class_index])

        # =========================
        # MODE DUMMY
        # =========================
        else:
            label_raw = random.choice(CLASS_LABELS)
            label = LABEL_TRANSLATE[label_raw]
            confidence = round(random.uniform(0.80, 0.99), 4)

        info = CLASS_INFO.get(label_raw, {})

        # =========================
        # SAVE HISTORY
        # =========================
        record = {
            'id': history_id_counter,
            'filename': filename,
            'label': label_raw,
            'confidence': confidence,
            'created_at': datetime.now()
        }

        history_store.append(record)
        history_id_counter += 1

        # =========================
        # RESPONSE
        # =========================
        return jsonify({
            'label': label,
            'confidence': confidence,
            'estimasi_panen': info.get('estimasi_panen'),
            'kondisi': info.get('kondisi'),
            'solusi': info.get('solusi'),
            'filename': filename
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

# =========================
# ERROR 404
# =========================
@app.errorhandler(404)
def not_found(e):
    return render_template('notfound.html'), 404

# =========================
# RUN
# =========================
if __name__ == '__main__':
    app.run(debug=True)