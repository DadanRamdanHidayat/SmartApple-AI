# =========================
# IMPORT LIBRARY
# =========================
from flask import Flask, render_template, request, jsonify, redirect, url_for
import os
from datetime import datetime
from werkzeug.utils import secure_filename
import random
import requests  # ← TAMBAH INI

# =========================
# MODE MODEL (ON / OFF)
# =========================
USE_MODEL = True

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

LABEL_TRANSLATE = {
    'Blotch_Apple': 'Bercak (Blotch)',
    'Normal_Apple': 'Apel Sehat',
    'Rot_Apple':    'Busuk',
    'Scab_Apple':   'Scab'
}

# =========================
# INFO PER KELAS
# =========================
CLASS_INFO = {
    'Normal_Apple': {
        'kondisi':       'Sehat',
        'estimasi_panen':'Siap dikonsumsi',
        'solusi':        'Tidak perlu tindakan khusus.'
    },
    'Blotch_Apple': {
        'kondisi':       'Terdapat bercak',
        'estimasi_panen':'Perlu seleksi',
        'solusi':        'Gunakan fungisida dan pisahkan buah.'
    },
    'Rot_Apple': {
        'kondisi':       'Busuk',
        'estimasi_panen':'Tidak layak',
        'solusi':        'Buang buah agar tidak menyebar.'
    },
    'Scab_Apple': {
        'kondisi':       'Terinfeksi scab',
        'estimasi_panen':'Perlu penanganan',
        'solusi':        'Perawatan lanjutan dengan fungisida.'
    }
}

# ==========================================================
# ✅ PERUBAHAN 1: API KEY GROQ — isi key kamu di sini
# ==========================================================
GROQ_API_KEY = "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"  # ← ganti dengan key kamu

# ==========================================================
# ✅ PERUBAHAN 2: FUNGSI REKOMENDASI AI — tambah fungsi baru
# ==========================================================
def get_ai_recommendation(label_raw, label_display, confidence, kondisi):

    prompt = f"""
Kamu adalah pakar pertanian apel dan penyakit tanaman.

Analisis hasil deteksi AI berikut:

Nama Penyakit : {label_display}
Kondisi Buah  : {kondisi}
Confidence AI : {confidence:.1f}%

Tugas kamu:
1. Jelaskan kondisi apel secara natural
2. Jelaskan tingkat bahaya penyakit
3. Berikan rekomendasi penanganan praktis
4. Jelaskan apakah apel masih layak dikonsumsi
5. Gunakan bahasa Indonesia yang ramah dan profesional
6. Maksimal 5 kalimat
7. Jangan gunakan format poin
8. Jangan gunakan pembuka seperti:
   - "Berikut hasil analisis"
   - "Tentu"
   - "Baik"

Buat jawaban terasa seperti konsultasi ahli sungguhan.
"""
    try:
        resp = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {GROQ_API_KEY}",
                "Content-Type":  "application/json"
            },
            json={
                "model":       "llama-3.3-70b-versatile",
                "messages":    [{"role": "user", "content": prompt}],
                "max_tokens":  200,
                "temperature": 0.7
            },
            timeout=8  # maksimal tunggu 8 detik
        )
        resp.raise_for_status()
        return resp.json()['choices'][0]['message']['content'].strip()

    except Exception as e:
        # Jika API error/timeout → pakai teks fallback
        print(f"[Groq API error] {e}")
        return CLASS_INFO.get(label_raw, {}).get('solusi', 'Tidak ada rekomendasi.')


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
        'total':  len(history_store),
        'normal': sum(1 for h in history_store if h['label'] == 'Normal_Apple'),
        'blotch': sum(1 for h in history_store if h['label'] == 'Blotch_Apple'),
        'rot':    sum(1 for h in history_store if h['label'] == 'Rot_Apple'),
        'scab':   sum(1 for h in history_store if h['label'] == 'Scab_Apple'),
    }
    return render_template('historypage.html', histories=history_store[::-1], stats=stats)

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
    img_array = keras_image.img_to_array(img) / 255.0
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
            prediction  = model.predict(img)[0]
            class_index = int(np.argmax(prediction))
            label_raw   = CLASS_LABELS[class_index]
            label       = LABEL_TRANSLATE.get(label_raw, label_raw)
            confidence  = float(prediction[class_index])

        # =========================
        # MODE DUMMY
        # =========================
        else:
            label_raw  = random.choice(CLASS_LABELS)
            label      = LABEL_TRANSLATE[label_raw]
            confidence = round(random.uniform(0.80, 0.99), 4)

        info = CLASS_INFO.get(label_raw, {})

        # ================================================================
        # ✅ PERUBAHAN 3: PANGGIL AI untuk rekomendasi, bukan hardcoded
        # ================================================================
        solusi = get_ai_recommendation(
            label_raw   = label_raw,
            label_display = label,
            confidence  = confidence * 100,
            kondisi     = info.get('kondisi', '')
        )

        # =========================
        # SAVE HISTORY
        # =========================
        record = {
            'id':         history_id_counter,
            'filename':   filename,
            'label':      label_raw,
            'confidence': confidence,
            'created_at': datetime.now()
        }
        history_store.append(record)
        history_id_counter += 1

        # =========================
        # RESPONSE
        # =========================
        return jsonify({
            'label':          label,
            'confidence':     confidence,
            'estimasi_panen': info.get('estimasi_panen'),
            'kondisi':        info.get('kondisi'),
            'solusi':         solusi,   # ← sekarang dari AI
            'filename':       filename
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
