import os
from flask import Flask, render_template, request, redirect, url_for
from werkzeug.utils import secure_filename
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.efficientnet import preprocess_input
from PIL import Image
import requests
import hashlib
from flask import jsonify

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
DATASET_DIR = os.path.join(BASE_DIR, 'Dataset')
NOTEBOOK_MODEL = os.path.join(BASE_DIR, 'notebooks', 'best_leaf_model.h5')
MODEL_PATH = NOTEBOOK_MODEL if os.path.exists(NOTEBOOK_MODEL) else os.path.join(DATASET_DIR, 'rubber_leaf_efficientnet_model.h5')
UPLOAD_FOLDER = os.path.join(os.path.dirname(__file__), 'static', 'uploads')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load model once at startup
model = load_model(MODEL_PATH)

# If you have a specific mapping from class name to index, use it here so predictions match that mapping
# Provided mapping:
CLASS_MAP = {
    'Abnormal': 0,
    'Anthracnose': 1,
    'Black_Spot': 2,
    'Dry_Leaf': 3,
    'Healthy': 4,
    'Leaf_Blight': 5,
    'Leaf_Spot': 6,
    'Powdery_Mildew': 7,
}

# Build class_labels ordered by index according to CLASS_MAP
class_labels = [label for label, _ in sorted(CLASS_MAP.items(), key=lambda kv: kv[1])]

def prepare_image(image_path, target_size=(224,224)):
    img = Image.open(image_path).convert('RGB')
    img = img.resize(target_size)
    arr = np.array(img)
    arr = preprocess_input(arr)
    arr = np.expand_dims(arr, axis=0)
    return arr

@app.route('/', methods=['GET', 'POST'])
def index():
    result = None
    probabilities = None
    image_url = None
    if request.method == 'POST':
        # If frontend provided a server-fetched filename, use it; otherwise use uploaded file
        fetched = request.form.get('fetched')
        if fetched:
            # sanitize and ensure only basename
            fetched_name = os.path.basename(fetched)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], fetched_name)
            if not os.path.exists(filepath):
                return redirect(request.url)
            image_url = url_for('static', filename=f'uploads/{fetched_name}')
        else:
            if 'image' not in request.files:
                return redirect(request.url)
            file = request.files['image']
            if file.filename == '':
                return redirect(request.url)
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            image_url = url_for('static', filename=f'uploads/{filename}')

        x = prepare_image(filepath)
        preds = model.predict(x)[0]
        probs = (preds * 100).tolist()

        # Map label -> probability (using defined class_labels order)
        probabilities_map = {label: float(probs[idx]) for idx, label in enumerate(class_labels)}

        # Also provide index -> probability mapping
        probabilities_by_index = {CLASS_MAP[label]: float(probs[idx]) for idx, label in enumerate(class_labels)}

        # For display: sort labels by probability descending
        paired_sorted = sorted(probabilities_map.items(), key=lambda x: x[1], reverse=True)

        result = paired_sorted[0][0]
        probabilities = paired_sorted

    return render_template('index.html', result=result, probabilities=probabilities, image_url=image_url)


@app.route('/fetch_url', methods=['POST'])
def fetch_url():
    data = request.get_json() or {}
    url = data.get('url')
    if not url:
        return jsonify({'ok': False, 'error': 'No URL provided'}), 400

    # basic validation
    if not (url.startswith('http://') or url.startswith('https://')):
        return jsonify({'ok': False, 'error': 'URL must start with http:// or https://'}), 400

    try:
        headers = { 'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0 Safari/537.36' }
        resp = requests.get(url, stream=True, timeout=15, headers=headers)
        resp.raise_for_status()
    except Exception as e:
        # log server-side for debugging
        print('fetch_url error:', repr(e))
        msg = str(e)
        return jsonify({'ok': False, 'error': f'Failed to fetch URL: {msg}'}), 400

    content_type = resp.headers.get('content-type', '')
    if not content_type.startswith('image/'):
        return jsonify({'ok': False, 'error': 'URL did not return an image'}), 400

    # limit size to 5MB
    max_bytes = 5 * 1024 * 1024
    total = 0
    hasher = hashlib.sha1()
    chunks = []
    for chunk in resp.iter_content(8192):
        if not chunk:
            break
        total += len(chunk)
        if total > max_bytes:
            return jsonify({'ok': False, 'error': 'Image too large (over 5MB)'}), 400
        hasher.update(chunk)
        chunks.append(chunk)

    ext = '.jpg'
    if 'png' in content_type:
        ext = '.png'
    elif 'jpeg' in content_type:
        ext = '.jpg'
    elif 'gif' in content_type:
        ext = '.gif'

    filename = hasher.hexdigest() + ext
    safe_name = secure_filename(filename)
    out_path = os.path.join(app.config['UPLOAD_FOLDER'], safe_name)

    with open(out_path, 'wb') as f:
        for c in chunks:
            f.write(c)

    file_url = url_for('static', filename=f'uploads/{safe_name}')
    return jsonify({'ok': True, 'filename': safe_name, 'url': file_url})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
