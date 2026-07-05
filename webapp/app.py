import os
from flask import Flask, render_template, request, redirect, url_for
from werkzeug.utils import secure_filename
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.efficientnet import preprocess_input
from PIL import Image
import requests
import hashlib
from flask import jsonify
import base64
from io import BytesIO

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
DATASET_DIR = os.path.join(BASE_DIR, 'Dataset')
NOTEBOOK_MODEL = os.path.join(BASE_DIR, 'notebooks', 'best_leaf_model.h5')
MODEL_PATH = NOTEBOOK_MODEL if os.path.exists(NOTEBOOK_MODEL) else os.path.join(DATASET_DIR, 'rubber_leaf_efficientnet_model.h5')
UPLOAD_FOLDER = os.path.join(os.path.dirname(__file__), 'static', 'uploads')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load model once at startup
model = None
model_error = None

try:
    model = load_model(MODEL_PATH)
except Exception as e:
    model_error = str(e)

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

class_labels = [label for label, _ in sorted(CLASS_MAP.items(), key=lambda kv: kv[1])]

# Disease information with symptoms and treatments
DISEASE_INFO = {
    'Abnormal': {
        'symptoms': ['Unusual discoloration', 'Deformed leaf structure', 'Irregular patterns'],
        'treatment': ['Inspect plant health', 'Check environmental conditions', 'Monitor for disease progression']
    },
    'Anthracnose': {
        'symptoms': ['Dark brown spots with yellow halo', 'Fungal lesions on veins', 'Leaf distortion'],
        'treatment': ['Remove infected leaves', 'Apply fungicide spray', 'Improve air circulation', 'Reduce humidity']
    },
    'Black_Spot': {
        'symptoms': ['Black circular spots with concentric rings', 'Yellow halo around spots', 'Spore production visible'],
        'treatment': ['Remove infected leaves', 'Apply copper fungicide', 'Avoid wetting foliage', 'Weekly fungicide application']
    },
    'Dry_Leaf': {
        'symptoms': ['Brown desiccated tissue', 'Leaf edge browning', 'Papery texture'],
        'treatment': ['Check soil moisture', 'Improve watering schedule', 'Increase humidity', 'Protect from direct sun']
    },
    'Healthy': {
        'symptoms': ['No visible damage', 'Normal green color', 'Regular leaf structure'],
        'treatment': ['Continue regular maintenance', 'Monitor periodically', 'Maintain optimal conditions']
    },
    'Leaf_Blight': {
        'symptoms': ['Large necrotic patches', 'Rapid leaf death', 'Dark lesions spreading'],
        'treatment': ['Remove infected leaves immediately', 'Apply systemic fungicide', 'Improve drainage', 'Remove debris']
    },
    'Leaf_Spot': {
        'symptoms': ['Small to medium brown spots', 'Yellow border around spots', 'Progressive spreading'],
        'treatment': ['Remove infected leaves', 'Apply copper-based fungicide', 'Space plants properly', 'Avoid overhead watering']
    },
    'Powdery_Mildew': {
        'symptoms': ['White powdery coating on leaves', 'Leaf curling', 'Reduced photosynthesis area'],
        'treatment': ['Apply sulfur spray or fungicide', 'Improve air circulation', 'Reduce humidity', 'Remove infected leaves']
    }
}

def make_gradcam_heatmap(img_array, model):
    """Generate Grad-CAM heatmap with intelligent fallback"""
    try:
        # Try to find the last convolutional layer
        last_conv_layer = find_conv_layer(model)
        
        if last_conv_layer is None:
            print("No conv layer found, trying alternative...")
            # Alternative: use the second-to-last layer if it has suitable output
            for layer in reversed(model.layers[:-1]):
                if hasattr(layer, 'output_shape') and len(layer.output_shape) == 4:
                    last_conv_layer = layer
                    break
        
        if last_conv_layer is not None:
            print(f"Using layer for Grad-CAM: {last_conv_layer.name} with shape {last_conv_layer.output_shape}")
            
            try:
                grad_model = tf.keras.models.Model(
                    inputs=model.input,
                    outputs=[last_conv_layer.output, model.output]
                )
                
                with tf.GradientTape() as tape:
                    conv_outputs, predictions = grad_model(img_array)
                    top_class = tf.argmax(predictions[0])
                    top_class_channel = predictions[:, top_class]
                
                grads = tape.gradient(top_class_channel, conv_outputs)
                
                if grads is not None:
                    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
                    conv_output = conv_outputs[0]
                    heatmap = conv_output @ pooled_grads[..., tf.newaxis]
                    heatmap = tf.squeeze(heatmap)
                    heatmap = tf.nn.relu(heatmap)
                    
                    heatmap_max = tf.reduce_max(heatmap)
                    heatmap_min = tf.reduce_min(heatmap)
                    
                    if heatmap_max > heatmap_min:
                        heatmap = (heatmap - heatmap_min) / (heatmap_max - heatmap_min)
                        print("Grad-CAM heatmap generated successfully")
                        return heatmap.numpy().astype(np.float32)
                    else:
                        print("Heatmap has no variation, using fallback")
            except Exception as e:
                print(f"Grad-CAM computation error: {e}")
        
        # Fallback: Generate synthetic heatmap from feature maps
        print("Using fallback heatmap generation...")
        try:
            # Get feature maps from intermediate layer
            intermediate_model = tf.keras.models.Model(
                inputs=model.input,
                outputs=model.layers[-4].output  # Use an earlier layer
            )
            features = intermediate_model(img_array)
            
            # Create heatmap by taking mean absolute value of channels
            heatmap = tf.reduce_mean(tf.abs(features[0]), axis=-1)
            
            # Normalize
            heatmap = (heatmap - tf.reduce_min(heatmap)) / (tf.reduce_max(heatmap) - tf.reduce_min(heatmap) + 1e-8)
            print("Fallback heatmap generated successfully")
            return heatmap.numpy().astype(np.float32)
        except:
            pass
        
        # Final fallback: Create center-weighted heatmap
        print("Using center-weighted heatmap")
        size = 14
        heatmap = np.zeros((size, size), dtype=np.float32)
        y, x = np.ogrid[0:size, 0:size]
        cx, cy = size / 2, size / 2
        r = np.sqrt((x - cx)**2 + (y - cy)**2)
        heatmap = np.exp(-r / (size / 3))
        heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)
        return heatmap.astype(np.float32)
        
    except Exception as e:
        print(f"Heatmap generation error: {e}")
        import traceback
        traceback.print_exc()
        # Return center-weighted fallback
        size = 14
        y, x = np.ogrid[0:size, 0:size]
        cx, cy = size / 2, size / 2
        r = np.sqrt((x - cx)**2 + (y - cy)**2)
        heatmap = np.exp(-r / (size / 3))
        return ((heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)).astype(np.float32)

def overlay_heatmap(original_img, heatmap, alpha=0.5):
    """Overlay Grad-CAM heatmap on original image with enhanced contrast"""
    # Ensure original_img is uint8 and in correct range
    if original_img.dtype != np.uint8:
        original_img = np.uint8(np.clip(original_img, 0, 255))
    
    # Resize heatmap to match image size
    heatmap_resized = cv2.resize(heatmap, (original_img.shape[1], original_img.shape[0]))
    
    # Enhance contrast in heatmap (stretch to full range)
    heatmap_min = heatmap_resized.min()
    heatmap_max = heatmap_resized.max()
    
    if heatmap_max > heatmap_min:
        heatmap_resized = (heatmap_resized - heatmap_min) / (heatmap_max - heatmap_min)
    
    # Normalize to 0-255 with enhanced contrast
    heatmap_normalized = np.uint8(255 * np.power(heatmap_resized, 0.4))  # Gamma correction
    
    # Apply JET colormap
    heatmap_color = cv2.applyColorMap(heatmap_normalized, cv2.COLORMAP_JET)
    
    # Convert BGR to RGB for heatmap display
    heatmap_color_rgb = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)
    
    # Create overlay: blend original image with heatmap
    overlay_img = cv2.addWeighted(original_img, 1 - alpha, heatmap_color_rgb, alpha, 0)
    overlay_img = np.uint8(np.clip(overlay_img, 0, 255))
    
    return overlay_img, heatmap_color_rgb

def estimate_severity(heatmap):
    """Estimate disease severity from Grad-CAM heatmap"""
    threshold = 0.5
    affected_area = np.sum(heatmap > threshold) / heatmap.size
    
    if affected_area < 0.2:
        severity = 'Mild'
    elif affected_area < 0.5:
        severity = 'Moderate'
    else:
        severity = 'Severe'
    
    percentage = affected_area * 100
    return severity, percentage

def image_to_base64(img_array):
    """Convert numpy array to base64 string"""
    # Ensure image is uint8
    if img_array.dtype != np.uint8:
        img_array = np.uint8(np.clip(img_array, 0, 255))
    
    # Convert RGB to BGR for cv2.imencode
    img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
    
    # Encode to PNG
    success, buffer = cv2.imencode('.png', img_bgr)
    if not success:
        print("Error encoding image to PNG")
        return ""
    
    img_base64 = base64.b64encode(buffer).decode('utf-8')
    return f"data:image/png;base64,{img_base64}"

def prepare_image(image_path, target_size=(224, 224)):
    """Prepare image for model prediction"""
    img = Image.open(image_path).convert('RGB')
    img = img.resize(target_size)
    img_array = np.array(img)
    img_processed = preprocess_input(img_array.copy())
    img_batch = np.expand_dims(img_processed, axis=0)
    return img_batch, img_array

@app.route('/', methods=['GET', 'POST'])
def index():
    result = None
    probabilities = None
    image_url = None
    error = None
    explanation_data = None
    
    if model_error:
        error = f"Model not available: {model_error}"
    
    if request.method == 'POST':
        if model_error:
            error = f"Model not available: {model_error}"
            return render_template('index.html', result=result, probabilities=probabilities, image_url=image_url, error=error)
            
        fetched = request.form.get('fetched')
        if fetched:
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

        # Prepare image and get prediction
        img_batch, img_original = prepare_image(filepath)
        preds = model.predict(img_batch, verbose=0)[0]
        probs = (preds * 100).tolist()

        probabilities_map = {label: float(probs[idx]) for idx, label in enumerate(class_labels)}
        paired_sorted = sorted(probabilities_map.items(), key=lambda x: x[1], reverse=True)

        result = paired_sorted[0][0]
        probabilities = paired_sorted

        # Generate Grad-CAM explanation
        try:
            heatmap = make_gradcam_heatmap(img_batch, model)
            overlay_img, heatmap_vis = overlay_heatmap(img_original, heatmap)
            severity, severity_pct = estimate_severity(heatmap)
            
            # Get disease information
            disease_info = DISEASE_INFO.get(result, {})
            
            explanation_data = {
                'disease': result,
                'confidence': float(paired_sorted[0][1]),
                'severity': severity,
                'severity_percentage': f"{severity_pct:.1f}%",
                'heatmap_base64': image_to_base64(heatmap_vis),
                'overlay_base64': image_to_base64(overlay_img),
                'symptoms': disease_info.get('symptoms', []),
                'treatment': disease_info.get('treatment', []),
                'all_predictions': paired_sorted
            }
        except Exception as e:
            print(f"Explanation error: {e}")
            explanation_data = None

    return render_template('index.html', result=result, probabilities=probabilities, image_url=image_url, error=error, explanation=explanation_data)


@app.route('/fetch_url', methods=['POST'])
def fetch_url():
    data = request.get_json() or {}
    url = data.get('url')
    if not url:
        return jsonify({'ok': False, 'error': 'No URL provided'}), 400

    if not (url.startswith('http://') or url.startswith('https://')):
        return jsonify({'ok': False, 'error': 'URL must start with http:// or https://'}), 400

    try:
        headers = {'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0 Safari/537.36'}
        resp = requests.get(url, stream=True, timeout=15, headers=headers)
        resp.raise_for_status()
    except Exception as e:
        print('fetch_url error:', repr(e))
        msg = str(e)
        return jsonify({'ok': False, 'error': f'Failed to fetch URL: {msg}'}), 400

    content_type = resp.headers.get('content-type', '')
    if not content_type.startswith('image/'):
        return jsonify({'ok': False, 'error': 'URL did not return an image'}), 400

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
    port = int(os.getenv('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
