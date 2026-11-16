"""Shared helpers used by main.py, train.py and the notebooks:
the 38 class names, finding/loading the trained model, and predicting one image."""
from pathlib import Path

import numpy as np

IMG_SIZE = 128  # the model was trained on 128 x 128 images

# The trained model is looked for under either of these file names.
MODEL_FILE_NAMES = ("trained_plant_disease_model.keras", "trained_model.keras")

# Order in which the model outputs its 38 classes (alphabetical folder order of the dataset).
CLASS_NAMES = [
    'Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy',
    'Blueberry___healthy', 'Cherry_(including_sour)___Powdery_mildew',
    'Cherry_(including_sour)___healthy', 'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot',
    'Corn_(maize)___Common_rust_', 'Corn_(maize)___Northern_Leaf_Blight', 'Corn_(maize)___healthy',
    'Grape___Black_rot', 'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)',
    'Grape___healthy', 'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot',
    'Peach___healthy', 'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy',
    'Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy',
    'Raspberry___healthy', 'Soybean___healthy', 'Squash___Powdery_mildew',
    'Strawberry___Leaf_scorch', 'Strawberry___healthy', 'Tomato___Bacterial_spot',
    'Tomato___Early_blight', 'Tomato___Late_blight', 'Tomato___Leaf_Mold',
    'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites Two-spotted_spider_mite',
    'Tomato___Target_Spot', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus',
    'Tomato___healthy',
]

HERE = Path(__file__).resolve().parent


class ModelProblem(Exception):
    """Something is wrong with the model file; the message says what to do."""


def find_model_file(folder=HERE):
    for name in MODEL_FILE_NAMES:
        path = Path(folder) / name
        if path.exists():
            return path
    return None


def load_model(folder=HERE):
    path = find_model_file(folder)
    if path is None:
        raise ModelProblem(
            "No trained model found. Put the model file in the same folder as main.py. "
            "It must be named 'trained_plant_disease_model.keras' (or 'trained_model.keras'). "
            "Get it by downloading it, or train it yourself with train.py (see README.md).")
    import tensorflow as tf
    try:
        model = tf.keras.models.load_model(path, compile=False)   # prediction only, no training state needed
    except Exception as e:  # noqa: BLE001 - show the reason to the user
        raise ModelProblem(
            f"Found {path.name} but could not load it:\n{e}\n\n"
            "A common cause is a model saved with a newer TensorFlow than the one installed.") from e
    if int(model.output_shape[-1]) != len(CLASS_NAMES):
        raise ModelProblem(
            f"{path.name} has {model.output_shape[-1]} outputs, but this project expects "
            f"{len(CLASS_NAMES)} classes (the 38-class PlantVillage model).")
    return model


def prepare_image(image):
    """PIL image (any size/mode) -> array of shape (1, 128, 128, 3), pixel values 0..255
    (the model contains no rescaling layer, so it expects raw 0..255 values)."""
    image = image.convert("RGB").resize((IMG_SIZE, IMG_SIZE))
    return np.asarray(image, dtype="float32")[None]


def predict(model, image, top=3):
    """Return [(class_name, probability 0..1), ...] best first."""
    probs = model.predict(prepare_image(image), verbose=0)[0]
    order = np.argsort(probs)[::-1][:top]
    return [(CLASS_NAMES[int(i)], float(probs[i])) for i in order]


def pretty(class_name):
    """'Tomato___Early_blight' -> ('Tomato', 'Early blight')."""
    plant, _, disease = class_name.partition("___")
    clean = lambda s: " ".join(s.replace("_", " ").split())
    return clean(plant), clean(disease)


def is_healthy(class_name):
    return class_name.lower().endswith("healthy")
