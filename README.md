<div align="center">

![YOLOv8](https://img.shields.io/badge/YOLOv8-Object%20Detection-blueviolet?style=for-the-badge&logo=python&logoColor=white)
![Flask](https://img.shields.io/badge/Flask-REST%20API-000000?style=for-the-badge&logo=flask&logoColor=white)
![Python](https://img.shields.io/badge/Python-3.8%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Status](https://img.shields.io/badge/Status-Live%20%26%20Working-brightgreen?style=for-the-badge)

# 🔍 YOLO Object Detection Web App

**Upload any image and get real-time object detection powered by YOLOv8 Nano — served through a Flask REST API with a clean dark-theme web interface.**

</div>

---

## 📸 Demo

> Upload an image → Click **Detect Objects** → See annotated results with bounding boxes and confidence scores instantly.

---

## ✨ Features

- 🖼️ **Drag & drop** image upload — no setup needed
- 🤖 Real-time inference using a **locally trained YOLOv8n model**
- 🎨 Annotated output with **color-coded bounding boxes**
- 📊 Detection table showing **object name, confidence %, and coordinates**
- ⚡ Live summary stats — **total objects**, **avg confidence**, **top class**

---

## 🗂️ Project Structure

```
yolo-object-detection/
│
├── 📁 runs/              # YOLOv8 training outputs & results
├── 📁 static/            # CSS, JS, and static assets
├── 📁 templates/         # Flask HTML templates
│
├── api.py                # ✅ Main Flask app — run this to start
├── best.pt               # 🧠 Custom trained YOLOv8 model weights
├── yolov8n.pt            # 🧠 Base YOLOv8 Nano pretrained weights
├── test.py               # 🧪 Standalone model test script
├── test_image.jpg        # 🖼️  Sample image for testing
├── bus.jpg               # 🖼️  COCO benchmark sample image
└── README.md             # 📄 You are here
```

---

## ⚡ Quick Start

**Step 1 — Clone the repository**

```bash
git clone https://github.com/zaidautomates/yolo-object-detection.git
```


**Step 2 — Move into the project folder**

```bash
cd yolo-object-detection
```

**Step 3 — Install required dependencies**

```bash
pip install ultralytics flask pillow numpy opencv-python
```

**Step 4 — Run the app**

```bash
python api.py
```

**Step 5 — Open in your browser**

```
http://localhost:5000
```

---

## 🔌 API Reference

### Endpoint

```
POST /detect
```

### Request

| Type | Value |
|------|-------|
| Content-Type | `multipart/form-data` |
| Body | Image file (JPG, PNG, WEBP) |

### Response (JSON)

```json
{
  "detections": [
    {
      "object": "person",
      "confidence": 90.3,
      "bounding_box": [140.14, 35.69, 437.68, 313.03]
    },
    {
      "object": "umbrella",
      "confidence": 54.1,
      "bounding_box": [309.49, 28.91, 470.46, 91.8]
    }
  ],
  "total_objects": 13,
  "avg_confidence": 59.4,
  "top_detection": "person"
}
```

---

## 🧠 Model Details

| Property | Value |
|----------|-------|
| Architecture | YOLOv8 Nano |
| Training Dataset | COCO128 |
| Custom Weights | `best.pt` |
| Input | Any image format |
| Framework | Ultralytics Python SDK |

---

## 👤 Author

**Zaid Ali**
AI Automation Developer · BS Computer Science · AWKUM


[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-0A66C2?style=flat-square&logo=linkedin&logoColor=white)](https://linkedin.com/in/zaidautomates)
[![GitHub](https://img.shields.io/badge/GitHub-Follow-181717?style=flat-square&logo=github&logoColor=white)](https://github.com/zaidautomates)



<div align="center">

⭐ **If this project helped you or impressed you — drop a star. It keeps the work going.**

*Built by Zaid Ali*

</div>
