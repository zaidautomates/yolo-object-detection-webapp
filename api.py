from flask import Flask, request, jsonify, render_template, send_from_directory
from ultralytics import YOLO
from PIL import Image
import io, os, base64, cv2, numpy as np

app = Flask(__name__, template_folder="templates", static_folder="static")

# Determine the safest model path (Task 8 & 9)
MODEL_PATH = "best.pt"
if not os.path.exists(MODEL_PATH):
    print("WARNING: best.pt not found! Ensure you run the Jupyter Notebook first.")

try:
    model = YOLO(MODEL_PATH)
except Exception:
    model = None

# ─────────────────────────────────────────────
#  Web UI  –  GET /
# ─────────────────────────────────────────────
@app.route("/", methods=["GET"])
def home():
    return render_template("index.html")

# ─────────────────────────────────────────────
#  Detection endpoint with annotated image
# ─────────────────────────────────────────────
@app.route("/detect", methods=["POST"])
def detect():
    if model is None:
        return jsonify({"error": "Model best.pt is missing. Train first!"}), 500

    if "image" not in request.files:
        return jsonify({"error": "No image file provided. Use parameter 'image'."}), 400

    file = request.files["image"]
    img_bytes = file.read()
    image = Image.open(io.BytesIO(img_bytes)).convert("RGB")

    # Run YOLOv8 inference
    results = model.predict(source=image, conf=0.25, verbose=False)

    # Convert PIL → OpenCV for drawing
    cv_img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

    detections = []
    # Curated colour palette for bounding boxes
    colors = [
        (0, 200, 255), (0, 255, 100), (255, 100, 0),
        (200, 0, 255), (255, 255, 0), (0, 150, 255),
        (255, 0, 150), (100, 255, 200), (255, 200, 100),
        (150, 0, 200),
    ]

    idx = 0
    for r in results:
        for box in r.boxes:
            cls_name = r.names[int(box.cls)]
            conf = float(box.conf)
            bbox = [round(c, 2) for c in box.xyxy[0].tolist()]
            x1, y1, x2, y2 = [int(c) for c in bbox]

            color = colors[idx % len(colors)]
            idx += 1

            # Draw filled rectangle behind label
            label = f"{cls_name} {conf*100:.1f}%"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
            cv2.rectangle(cv_img, (x1, y1 - th - 10), (x1 + tw + 6, y1), color, -1)
            cv2.putText(cv_img, label, (x1 + 3, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

            # Bounding box
            cv2.rectangle(cv_img, (x1, y1), (x2, y2), color, 3)

            detections.append({
                "class": cls_name,
                "confidence": round(conf, 4),
                "confidence_pct": f"{conf*100:.1f}%",
                "bbox": bbox,
            })

    # Encode annotated image → base64
    _, buf = cv2.imencode(".jpg", cv_img, [cv2.IMWRITE_JPEG_QUALITY, 92])
    b64_img = base64.b64encode(buf).decode("utf-8")

    return jsonify({
        "total_detections": len(detections),
        "detections": detections,
        "annotated_image": b64_img,
    })

# ─────────────────────────────────────────────
#  Legacy JSON-only endpoint (backward compat)
# ─────────────────────────────────────────────
@app.route("/predict", methods=["POST"])
def predict():
    if model is None:
        return jsonify({"error": "Model best.pt is missing. Train first!"}), 500

    if "image" not in request.files:
        return jsonify({"error": "No image passed parameter 'image' allowed"}), 400

    file = request.files["image"]
    image = Image.open(io.BytesIO(file.read())).convert("RGB")

    results = model.predict(source=image, conf=0.25, verbose=False)

    detections = []
    for r in results:
        for box in r.boxes:
            detections.append({
                "class": r.names[int(box.cls)],
                "confidence": round(float(box.conf), 4),
                "bbox": [round(i, 2) for i in box.xyxy[0].tolist()]
            })

    return jsonify({
        "total_detections": len(detections),
        "detections": detections
    })

if __name__ == "__main__":
    print("\n✅  YOLO Web UI running at  http://localhost:5000\n")
    app.run(host="0.0.0.0", port=5000, debug=False)
