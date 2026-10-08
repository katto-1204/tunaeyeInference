import json
import qrcode

SSID = "TunaRpi"
PASSWORD = ""
API_URL = "http://10.42.0.1:5000"
STREAM_URL = "http://10.42.0.1:8080/stream"

payload = {
    "ssid": SSID,
    "password": PASSWORD,
    "apiUrl": API_URL,
    "streamUrl": STREAM_URL,
}

data = json.dumps(payload)
img = qrcode.make(data)
img.save("qr.png")
print("QR code saved to qr.png")
print("Encoded payload:", data)
