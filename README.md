
# TunaEye — Raspberry Pi 5 Backend

Raspberry Pi 5 backend for TunaEye's automated tuna quality grading system. Provides USB camera streaming and TensorFlow Lite inference for the TunaEye Kiosk (Tablet PWA) and Mobile App (Expo).

## Folder Structure

```text
rpi-cam-demo/
├── model/             # TFLite grading models
├── pwa_build/         # Optional local PWA build
├── app.py             # Flask API (port 5000)
├── generate_qr.py     # Generates connection QR
├── qr.png             # QR code
└── README.md
```

ustreamer is installed separately in `~/ustreamer/`.

## Architecture

```text
TunaEye Kiosk (Tablet PWA)
TunaEye Mobile App (Expo)
           |
      TunaRpi Wi-Fi
           |
    Raspberry Pi 5
      10.42.0.1
           |
     +-----+-----+
     |           |
 ustreamer     Flask API
 Port 8080     Port 5000
     |           |
 USB Camera   TFLite AI
              Sashibo-Core
              Tail-Cut
```

- **Vercel:** Hosts the online kiosk frontend.
- **TunaRpi:** Provides local Wi-Fi connectivity.
- **ustreamer:** Handles USB camera streaming.
- **Flask:** Handles API requests and AI inference.
- **TensorFlow Lite:** Runs grading models on the Pi.

## Installation

```bash
sudo apt update
sudo apt install -y python3-pip python3-venv

cd ~/rpi-cam-demo
python3 -m venv --system-site-packages .venv
source .venv/bin/activate

pip install flask flask-cors requests pillow numpy "qrcode[pil]"
pip install tflite-runtime
```

If `tflite-runtime` is unavailable, install a compatible TensorFlow Lite interpreter package.

## Network Configuration

| Setting | Value |
|---------|-------|
| Hostname | TunaPi |
| Username | tunarpi |
| Hotspot SSID | TunaRpi |
| Raspberry Pi IP | 10.42.0.1 |
| Camera Port | 8080 |
| Flask Port | 5000 |

The hotspot supports local communication without internet access.

## Start Services

### 1. Start Camera Streaming

```bash
cd ~/ustreamer
./ustreamer --device=/dev/video0 --host=0.0.0.0 --port=8080
```

### 2. Start Flask API

Open another terminal:

```bash
cd ~/rpi-cam-demo
source .venv/bin/activate
python3 app.py
```

Both services must be running for camera-based inference.

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `:8080/stream` | Live USB camera |
| GET | `:8080/snapshot` | Capture camera frame |
| GET | `:5000/status` | Flask health check |
| POST | `:5000/grade` | Run TFLite inference |
| GET | `:5000/` | Serve local PWA if configured |

Base IP: `http://10.42.0.1`

## Camera and Inference Workflow

1. Kiosk or mobile app connects to TunaRpi.
2. Camera screen displays the ustreamer feed.
3. User initiates grading.
4. Flask retrieves the camera snapshot.
5. TensorFlow Lite runs the appropriate model.
6. Prediction returns to the kiosk or app.

The camera feed is embedded within the application. No external browser tab is required.

## Offline Operation

- Camera streaming operates locally.
- AI inference runs on Raspberry Pi 5.
- Internet is not required for local processing.
- The installed kiosk PWA retains its existing offline capabilities.
- Pending records can synchronize when internet returns, if synchronization is configured.

**Important:** The Vercel HTTPS PWA requires a browser-compatible secure connection to local Raspberry Pi services. Direct HTTP requests may be blocked. A trusted local HTTPS gateway or compatible local deployment must be configured.

## QR Code

Generate the connection QR:

```bash
python3 generate_qr.py
```

**Warning:** This overwrites `qr.png`. Back up the existing QR code before running.

## Deployment Notes

- Keep ustreamer and Flask running.
- Store TFLite models inside `model/`.
- Verify model preprocessing and class labels.
- Configure systemd for automatic startup.
- Keep inference and camera processing on the Raspberry Pi.
- Preserve the existing Vercel PWA and Expo application.

**Status:** Separate Sashibo-Core/Tail-Cut inference, secure PWA access, and systemd autostart require verification or implementation.
