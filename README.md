# Leaf Disease Predictor — Webapp

This folder contains a small Flask web application that loads a saved Keras model and provides a UI to upload or import leaf images and see predicted class probabilities.

Features
- Upload image file or drag & drop
- Import image via URL (server-side fetch to avoid CORS)
- Camera capture (mobile/desktop with a camera)
- Paste from clipboard
- Shows class probability percentages and animated progress bars

Quick start (Windows)

1. From the workspace root create and activate a virtual environment:

```powershell
python -m venv .venv
& ".venv\Scripts\Activate.ps1"
```

2. Install Python dependencies (run from repository root):

```powershell
pip install -r requirements.txt
```

3. Run the webapp:

```powershell
cd webapp
python app.py
```

4. Open the UI in your browser: http://127.0.0.1:5000/

Model files and class mapping
- The app prefers a model at `notebooks/best_leaf_model.h5` if present; otherwise it falls back to `Dataset/rubber_leaf_efficientnet_model.h5`.
- The app enforces the class mapping used for predictions:

```
{'Abnormal': 0,
 'Anthracnose': 1,
 'Black_Spot': 2,
 'Dry_Leaf': 3,
 'Healthy': 4,
 'Leaf_Blight': 5,
 'Leaf_Spot': 6,
 'Powdery_Mildew': 7}
```

File uploads
- Uploaded files and server-fetched URL images are saved to `webapp/static/uploads/` with SHA1-based filenames for server-fetched images.

Important endpoints
- `/` — GET serves the UI; POST receives a file upload or a `fetched` filename (server-fetched URL) and returns a page with predictions.
- `/fetch_url` — POST JSON `{'url': 'https://...'}`. Server fetches the image and returns `{ok: true, filename, url}` or an error. Limits: content-type must be `image/*`, max fetch size 5MB, 15s timeout.

Troubleshooting
- If the page shows no preview after importing from URL:
	- Check the server console (where you run `python app.py`) for errors printed by the `/fetch_url` handler.
	- Check the browser console for network errors or CORS messages.
- If model loading is slow or fails, ensure your Python environment has a compatible TensorFlow build and enough memory. See `requirements.txt` for packages.
- If you get `ImportError`/`ModuleNotFoundError`, confirm the virtual environment is activated and `pip install -r requirements.txt` completed successfully.

Testing flows
- Try each importer in the UI: file input, drag & drop, camera capture, URL import, and clipboard paste.
- For URL imports: paste a direct image URL, click Load Image, verify preview appears, then click `Upload & Predict`.

Extending and production
- For production use, run behind a WSGI server (Gunicorn/Waitress) and serve static files via a proper web server. Disable `debug=True` in `app.py`.
- Consider adding a cleanup policy for `static/uploads/` to prevent disk growth.

Support
- If you run into issues, copy the server traceback and the browser console log when reproducing the problem and open an issue or ask for help.

License / Attribution
- This project is provided as-is for demonstration and local testing.

