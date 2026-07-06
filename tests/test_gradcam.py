import numpy as np
import tensorflow as tf

from webapp import app as app_module


def test_make_gradcam_heatmap_returns_array_for_simple_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(224, 224, 3)),
        tf.keras.layers.Conv2D(4, 3, activation='relu'),
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(1, activation='sigmoid'),
    ])

    img_batch = np.random.rand(1, 224, 224, 3).astype(np.float32)
    heatmap = app_module.make_gradcam_heatmap(img_batch, model)

    assert heatmap.shape == (14, 14)
    assert np.isfinite(heatmap).all()
    assert heatmap.min() >= 0.0
    assert heatmap.max() <= 1.0
