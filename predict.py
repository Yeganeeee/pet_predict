""""Resuable inference helper for the Cats vs Dogs project

The saved model already contains MobileNetV2 processing, so images
must be passed as raw RGB pixels in the 0-255 range 

"""
import numpy as np 
import sys
from pathlib import Path

import json
import tensorflow as tf 
from PIL import Image


IMG_SIZE=160
PROJECT_DIR=Path(__file__).resolve().parent
MODEL_PATH=PROJECT_DIR/ "model" / "cats_dogs_transfer.keras"
CLASS_NAMES_PATH = PROJECT_DIR / "model" / "class_names.json"

def load_class_names(path=CLASS_NAMES_PATH):
    with open(path, encoding="UTF-8") as fhand:
        return json.load(fhand)

def load_model(path=MODEL_PATH):
    return tf.keras.models.load_model(path)

def process_pil_image(image, image_size=IMG_SIZE):
    image = image.convert("RGB").resize((image_size, image_size))
    array = np.asarray(image, dtype=np.float32)
    return np.expand_dims(array, axis=0)

def process_image(image_path, image_size=IMG_SIZE):
    with Image.open(image_path) as image:
        return process_pil_image(image, image_size)
def _predict_batch(model, class_names, batch):
    scores= model.predict(batch, verbose=0)[0]
    index=int(np.argmax(scores))
    return class_names[index], float(scores[index]), scores

def predict_image(model,class_names,image_path):
    return _predict_batch(model, class_names, process_image(image_path))
def predict_pil_image(model, class_names, image):
    return _predict_batch(model, class_names,process_pil_image(image))

if __name__=="__main__":
    if len(sys.argv) !=2:
        raise SystemExit("Usage: python predict.py path/to/image")

    model = load_model()
    class_names= load_class_names()

    label, confidence, scores = predict_image(model, class_names, sys.argv[1])

    print(f"Prediction:{label}")
    print(f"Confidence:{confidence}")
    print(f"Scores:{scores}")