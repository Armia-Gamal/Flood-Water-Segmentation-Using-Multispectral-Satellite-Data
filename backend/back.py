from flask import Flask, request, jsonify
import numpy as np
import tensorflow as tf
import cv2
import base64
import rasterio
from rasterio.io import MemoryFile

app = Flask(__name__)

model = tf.keras.models.load_model(
    r"d:\Study\projects\Flood-Water-Segmentation-Using-Multispectral-Satellite-Data\models\best_keras_model.h5"
)

IMG_SIZE = 128


def preprocess_image(file):

    file_bytes = file.read()

    with MemoryFile(file_bytes) as memfile:
        with memfile.open() as src:
            img = src.read()  

    img = np.transpose(img, (1,2,0)).astype("float32")

    for i in range(img.shape[2]):
        band = img[:,:,i]
        img[:,:,i] = (band - band.min()) / (band.max() - band.min() + 1e-6)

    return img


def encode_image(img):

    _, buffer = cv2.imencode(".png", img)
    return base64.b64encode(buffer).decode("utf-8")


@app.route("/")
def home():
    return jsonify({"message": "Flood Water Segmentation API is running"})


@app.route("/predict", methods=["POST"])
def predict():

    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    file = request.files["image"]

    try:

        img = preprocess_image(file)

        X = np.expand_dims(img, axis=0)

        pred = model.predict(X)

        pred_mask = (pred > 0.5).astype(np.uint8)[0,:,:,0] * 255

        pred_mask_rgb = cv2.cvtColor(pred_mask, cv2.COLOR_GRAY2BGR)

        rgb = img[:,:,1:4]

        rgb = (rgb - rgb.min()) / (rgb.max() - rgb.min() + 1e-6)
        rgb = (rgb * 255).astype(np.uint8)

        overlay = rgb.copy()
        overlay[pred_mask > 0] = [255,0,0]

        return jsonify({
            "image": encode_image(rgb),
            "pred_mask": encode_image(pred_mask_rgb),
            "overlay": encode_image(overlay)
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True)