# Object Detection — YOLOv3 / Faster R-CNN / CNN

A Python-based object detection project implementing and comparing three major detection architectures: **YOLOv3**, **Faster R-CNN**, and a custom **CNN classifier**, applied to static images using the COCO and VOC datasets.

---

## 📁 Project Structure

```
object-detection/
├── main.py               # Entry point — model selection & inference pipeline
├── YOLO_v3.py            # YOLOv3 detection logic
├── Faster_R_Cnn.py       # Faster R-CNN detection logic
├── CNN.py                # Custom CNN classifier
├── check_modules.py      # Dependency checker
├── coco.names            # COCO class labels (80 classes)
├── voc.names             # VOC class labels (20 classes)
├── voc.cfg               # VOC model config
├── yolov3.cfg            # YOLOv3 architecture config
├── requirements.txt      # Python dependencies
├── images/               # Input images for inference
└── README.md
```

> ⚠️ **Model weights are not included** due to file size. See [Download Weights](#-download-weights) below.

---

## ⚙️ Requirements

- Python 3.8+
- OpenCV
- TensorFlow / Keras
- NumPy

Install all dependencies:

```bash
pip install -r requirements.txt
```

---

## 📥 Download Weights

The following model files must be downloaded separately and placed in the project root:

| File | Size | Source |
|------|------|--------|
| `yolov3.weights` | ~242 MB | [Official YOLOv3](https://pjreddie.com/media/files/yolov3.weights) |
| `model.keras` | ~484 KB | Google Drive / Release |
| `model_V2.keras` | ~14 MB | Google Drive / Release |

> You can also find them in the [Releases](../../releases) section of this repository.

---

## 🚀 Usage

Run detection on a static image:

```bash
python main.py --model yolo --image images/sample.jpg
python main.py --model faster_rcnn --image images/sample.jpg
python main.py --model cnn --image images/sample.jpg
```

### Available arguments

| Argument | Values | Description |
|----------|--------|-------------|
| `--model` | `yolo`, `faster_rcnn`, `cnn` | Model to use for detection |
| `--image` | path/to/image | Input image path |
| `--conf` | float (default: 0.5) | Confidence threshold |
| `--output` | path/to/output | Save result image |

---

## 🧠 Models Overview

### YOLOv3
- Single-pass detection, high speed
- Uses `yolov3.cfg` + `yolov3.weights` + `coco.names`
- 80 COCO classes

### Faster R-CNN
- Two-stage detector: Region Proposal Network + classifier
- Higher accuracy, slower inference
- Uses `voc.cfg` + `voc.names`

### CNN (Custom)
- Image classifier based on Keras
- Trained model saved as `model.keras` / `model_V2.keras`
- Suitable for single-class or fine-grained classification

---

## 📊 Results Comparison

| Model | Speed | Accuracy | Use Case |
|-------|-------|----------|----------|
| YOLOv3 | ⚡ Fast | ✅ Good | General detection |
| Faster R-CNN | 🐢 Slow | 🎯 High | Precision tasks |
| CNN | ⚡ Fast | ⚠️ Limited | Classification |

---

## 🔧 Check Dependencies

```bash
python check_modules.py
```

---
