
import io
import logging
import os
import threading
import uuid
from pathlib import Path

import numpy as np
import requests
from PIL import Image, UnidentifiedImageError
from flask import Flask, Response, jsonify, request
from flask_cors import CORS
from ai_edge_litert.interpreter import Interpreter


# ============================================================
# TUNAEYE V2 — RASPBERRY PI 5 INFERENCE API
# Backend only. Frontends run in separate repositories.
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "model"

MODEL_PATHS = {
    "sashibocore": MODEL_DIR / "tunaeye_v2_sashibocore_fp32.tflite",
    "tailcut": MODEL_DIR / "tunaeye_v2_tailcut_fp32.tflite",
}

CLASSES = ["GRADE_A", "GRADE_B", "GRADE_C", "INVALID"]

CAMERA_BASE_URL = os.getenv(
    "TUNAEYE_CAMERA_URL",
    "http://127.0.0.1:8080",
).rstrip("/")

MAX_IMAGE_BYTES = 10 * 1024 * 1024

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = MAX_IMAGE_BYTES

# Development: allow kiosk/mobile clients on local hotspot.
# Restrict origins before exposing API to other networks.
CORS(app, resources={r"/*": {"origins": "*"}})

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("tunaeye")

models = {}
model_locks = {}


# ============================================================
# MODEL LOADING
# ============================================================

def load_models():
    for model_type, path in MODEL_PATHS.items():
        if not path.is_file():
            logger.error("Missing model: %s", path)
            continue

        try:
            interpreter = Interpreter(model_path=str(path))
            interpreter.allocate_tensors()

            input_info = interpreter.get_input_details()[0]
            output_info = interpreter.get_output_details()[0]

            if list(input_info["shape"]) != [1, 224, 224, 3]:
                raise ValueError("Unexpected input shape")

            if list(output_info["shape"]) != [1, 4]:
                raise ValueError("Unexpected output shape")

            if input_info["dtype"] != np.float32:
                raise ValueError("Expected float32 input")

            if output_info["dtype"] != np.float32:
                raise ValueError("Expected float32 output")

            models[model_type] = {
                "interpreter": interpreter,
                "input_index": input_info["index"],
                "output_index": output_info["index"],
            }

            model_locks[model_type] = threading.Lock()

            logger.info("Loaded %s: %s", model_type, path.name)

        except Exception:
            logger.exception("Failed loading model %s", model_type)


# ============================================================
# IMAGE PREPROCESSING — MATCH TUNAEYE V2 TRAINING
# ============================================================

def preprocess_image(image_bytes):
    with Image.open(io.BytesIO(image_bytes)) as image:
        image = image.convert("RGB")

        width, height = image.size
        side = min(width, height)

        left = (width - side) // 2
        top = (height - side) // 2

        image = image.crop(
            (left, top, left + side, top + side)
        )

        # Matches cv2.INTER_LINEAR for ordinary 8-bit RGB
        # resizing closely; verify against Colab reference.
        image = image.resize(
            (224, 224),
            resample=Image.Resampling.BILINEAR,
        )

        array = np.asarray(image, dtype=np.float32)

    # IMPORTANT:
    # MobileNetV3Small include_preprocessing=True.
    # Model expects RGB float32 in range 0–255.
    # DO NOT divide by 255.
    # DO NOT normalize to -1..1.

    return np.expand_dims(array, axis=0)


# ============================================================
# INFERENCE
# ============================================================

def run_inference(image_bytes, image_type):
    model = models[image_type]
    interpreter = model["interpreter"]

    tensor = preprocess_image(image_bytes)

    # TFLite interpreters must not be invoked concurrently.
    with model_locks[image_type]:
        interpreter.set_tensor(model["input_index"], tensor)
        interpreter.invoke()

        scores = interpreter.get_tensor(
            model["output_index"]
        )[0].copy()

    if scores.shape != (4,):
        raise ValueError("Unexpected model output shape")

    if not np.all(np.isfinite(scores)):
        raise ValueError("Non-finite model scores")

    if np.any(scores < -0.001) or np.any(scores > 1.001):
        raise ValueError("Model output is not probabilities")

    if not np.isclose(float(scores.sum()), 1.0, atol=0.01):
        raise ValueError("Probabilities do not sum to 1")

    index = int(np.argmax(scores))
    grade = CLASSES[index]
    confidence = float(scores[index])

    return {
        "grade": grade,
        "confidence": confidence,
        "scores": {
            label: float(scores[i])
            for i, label in enumerate(CLASSES)
        },
    }


# ============================================================
# HEALTH CHECK
# ============================================================

@app.get("/status")
def status():
    return jsonify({
        "status": "ok" if len(models) == 2 else "degraded",
        "service": "TunaEye V2 Inference API",
        "version": "2.0",
        "models": {
            name: name in models
            for name in MODEL_PATHS
        },
        "classes": CLASSES,
    })


# ============================================================
# GRADING API — EXACT KIOSK CONTRACT
# ============================================================

@app.post("/grade")
def grade():
    image_type = request.form.get(
        "image_type", ""
    ).strip().lower()

    if image_type not in MODEL_PATHS:
        return jsonify({
            "error": "invalid_image_type",
            "message": (
                "image_type must be 'sashibocore' "
                "or 'tailcut'."
            ),
        }), 400

    if image_type not in models:
        return jsonify({
            "error": "model_unavailable",
            "message": f"{image_type} model is not loaded.",
        }), 503

    uploaded = request.files.get("image")

    if uploaded is None:
        return jsonify({
            "error": "missing_image",
            "message": "Upload image using form field 'image'.",
        }), 400

    try:
        image_bytes = uploaded.read()

        if not image_bytes:
            return jsonify({
                "error": "empty_image",
                "message": "Uploaded image is empty.",
            }), 400

        if len(image_bytes) > MAX_IMAGE_BYTES:
            return jsonify({
                "error": "image_too_large",
                "message": "Maximum image size is 10 MB.",
            }), 413

        result = run_inference(image_bytes, image_type)

        inference_id = str(uuid.uuid4())
        capture_id = request.form.get("capture_id", "").strip()

        if not capture_id:
            capture_id = str(uuid.uuid4())

        return jsonify({
            "id": inference_id,
            "capture_id": capture_id,
            "image_type": image_type,
            "grade": result["grade"],
            "confidence": result["confidence"],
            "scores": result["scores"],
        }), 200

    except (UnidentifiedImageError, OSError, ValueError) as exc:
        return jsonify({
            "error": "invalid_image",
            "message": str(exc),
        }), 400

    except Exception:
        logger.exception("Inference failed")

        return jsonify({
            "error": "inference_failed",
            "message": "Unable to process image.",
        }), 500


# ============================================================
# OPTIONAL CAMERA PROXY
# ustreamer remains the camera server on port 8080.
# ============================================================

@app.get("/snapshot")
def snapshot():
    try:
        upstream = requests.get(
            f"{CAMERA_BASE_URL}/snapshot",
            timeout=5,
        )
        upstream.raise_for_status()

        return Response(
            upstream.content,
            content_type=upstream.headers.get(
                "Content-Type", "image/jpeg"
            ),
        )

    except requests.RequestException:
        return jsonify({
            "error": "camera_unavailable",
            "message": "Camera snapshot unavailable.",
        }), 503


@app.get("/stream")
def stream():
    try:
        upstream = requests.get(
            f"{CAMERA_BASE_URL}/stream",
            stream=True,
            timeout=(3, 15),
        )
        upstream.raise_for_status()

    except requests.RequestException:
        return jsonify({
            "error": "camera_unavailable",
            "message": "Camera stream unavailable.",
        }), 503

    def generate():
        try:
            for chunk in upstream.iter_content(
                chunk_size=16384
            ):
                if chunk:
                    yield chunk
        except requests.RequestException:
            logger.warning("Camera stream disconnected")
        finally:
            upstream.close()

    return Response(
        generate(),
        content_type=upstream.headers.get(
            "Content-Type",
            "multipart/x-mixed-replace; boundary=boundarydonotcross",
        ),
    )


# ============================================================
# ERRORS
# ============================================================

@app.errorhandler(413)
def image_too_large(error):
    return jsonify({
        "error": "image_too_large",
        "message": "Maximum request size is 10 MB.",
    }), 413


# ============================================================
# START SERVER
# ============================================================

load_models()

if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=5000,
        debug=False,
        threaded=True,
    )
