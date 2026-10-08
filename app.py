import io
import os
import numpy as np
import requests
from PIL import Image
from flask import Flask, jsonify
from flask_cors import CORS

try:
    import tflite_runtime.interpreter as tflite
    Interpreter = tflite.Interpreter
except ImportError:
    import tensorflow as tf
    Interpreter = tf.lite.Interpreter

app = Flask(__name__, static_folder="pwa_build", static_url_path="")
CORS(app)

MODEL_PATH = "model/tuna_grader.tflite"
interpreter = None
if os.path.exists(MODEL_PATH):
    interpreter = Interpreter(model_path=MODEL_PATH)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

def grab_frame_for_inference():
    resp = requests.get("http://localhost:8080/snapshot", timeout=3)
    return resp.content

def run_inference(jpeg_bytes):
    img = Image.open(io.BytesIO(jpeg_bytes)).convert("RGB")
    img = img.resize((224, 224))  # must match your training input size
    arr = np.array(img, dtype=np.float32) / 255.0  # must match your training normalization
    arr = np.expand_dims(arr, axis=0)
    interpreter.set_tensor(input_details[0]["index"], arr)
    interpreter.invoke()
    return interpreter.get_tensor(output_details[0]["index"])

@app.route("/")
def serve_pwa():
    return app.send_static_file("index.html")

@app.route("/status")
def status():
    return jsonify({"status": "ok", "message": "Raspberry Pi API is reachable"})

@app.route("/grade", methods=["POST"])
def grade():
    if interpreter is None:
        return jsonify({"error": "model not installed"}), 503
    result = run_inference(grab_frame_for_inference())
    return jsonify({"grade": result.tolist()})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
