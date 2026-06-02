import os
import io
import base64
import numpy as np
import cv2
import joblib
from flask import (Flask, render_template, request, redirect,
                   url_for, flash, jsonify)
from werkzeug.utils import secure_filename
from ml_model import predict_image, train_model, CLASSES, MODEL_DIR, STATIC_DIR

# ─────────────────────────────────────────────────────────────────────────────
app = Flask(__name__)
app.secret_key = 'knn-fruit-secret-2024'

UPLOAD_FOLDER   = os.path.join('static', 'uploads')
ALLOWED_EXTS    = {'png', 'jpg', 'jpeg', 'webp'}
app.config['UPLOAD_FOLDER']  = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 10 * 1024 * 1024   # 10 MB

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(STATIC_DIR,    exist_ok=True)


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTS


def model_trained():
    return os.path.exists(os.path.join(MODEL_DIR, 'knn_model.pkl'))


# ─────────────────────────────────────────────────────────────────────────────
# Routes
# ─────────────────────────────────────────────────────────────────────────────
@app.route('/')
def index():
    trained = model_trained()
    metrics = None
    if trained:
        m_path = os.path.join(MODEL_DIR, 'metrics.pkl')
        if os.path.exists(m_path):
            metrics = joblib.load(m_path)
    return render_template('index.html', trained=trained, metrics=metrics)


@app.route('/train', methods=['POST'])
def train():
    try:
        metrics = train_model()
        flash(f'✅ Model berhasil dilatih! Akurasi validasi: {metrics["accuracy"]*100:.2f}%', 'success')
    except Exception as e:
        flash(f'❌ Gagal melatih model: {str(e)}', 'danger')
    return redirect(url_for('index'))


@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if not model_trained():
        flash('⚠️ Model belum dilatih. Latih model terlebih dahulu.', 'warning')
        return redirect(url_for('index'))

    result = None
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('Tidak ada file yang dipilih.', 'danger')
            return redirect(request.url)

        file = request.files['file']
        if file.filename == '':
            flash('Tidak ada file yang dipilih.', 'danger')
            return redirect(request.url)

        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            # Read image with OpenCV
            img = cv2.imread(filepath)
            if img is None:
                flash('Gagal membaca gambar. Coba file lain.', 'danger')
                return redirect(request.url)

            try:
                pred_cls, confidence, probs = predict_image(img)

                # Prepare chart data (sorted by prob)
                sorted_probs = sorted(probs.items(), key=lambda x: x[1], reverse=True)

                result = {
                    'predicted_class' : pred_cls,
                    'confidence'      : round(confidence * 100, 2),
                    'probs'           : [(cls, round(p * 100, 2)) for cls, p in sorted_probs],
                    'image_path'      : filepath.replace('\\', '/'),
                    'image_url'       : url_for('static', filename=f'uploads/{filename}'),
                }
            except Exception as e:
                flash(f'❌ Prediksi gagal: {str(e)}', 'danger')
                return redirect(request.url)
        else:
            flash('Format file tidak didukung. Gunakan PNG, JPG, atau JPEG.', 'danger')
            return redirect(request.url)

    return render_template('predict.html', result=result)


@app.route('/metrics')
def metrics_page():
    if not model_trained():
        flash('⚠️ Model belum dilatih.', 'warning')
        return redirect(url_for('index'))

    m_path = os.path.join(MODEL_DIR, 'metrics.pkl')
    metrics = joblib.load(m_path) if os.path.exists(m_path) else {}
    return render_template('metrics.html', metrics=metrics)


@app.route('/about')
def about():
    return render_template('about.html')


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
