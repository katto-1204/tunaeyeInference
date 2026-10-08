# rpi-cam-demo

Raspberry Pi tuna grading demo. Flask API plus PWA front end.

## Folder layout

- model/ : put tuna_grader.tflite here
- pwa_build/ : put the built PWA files here (index.html at the top level)
- app.py : Flask server (port 5000)
- generate_qr.py : makes a new qr.png. This OVERWRITES the existing qr.png
- qr.png : the team's own QR code

## Install

    pip install flask flask-cors requests pillow numpy qrcode
    pip install tflite-runtime

If tflite-runtime is not available, install tensorflow instead.

## Run

    python3 app.py

The camera stream must be running on port 8080 with a /snapshot endpoint.

## Endpoints

- GET / : serves the PWA
- GET /status : health check
- POST /grade : grabs a frame from localhost:8080/snapshot and runs the model
