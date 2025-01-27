# Real-Time Traffic Sign Recognition with YOLOv5

This repository provides a real-time traffic sign detection and classification system based on the YOLOv5 architecture. It uses a dataset of 41 different types of traffic signs (provided by Fırat University Big Data and AI Laboratory) and supports both local (Flask-based) and remote (WebRTC-based) deployment scenarios. The system offers additional features such as text-to-speech (TTS) alerts and bounding box visualizations.

---

## Table of Contents
1. [Overview](#overview)
2. [Dataset](#dataset)
3. [Annotation Conversion](#annotation-conversion)
4. [Model Training](#model-training)
5. [Training Results](#training-results)
6. [Local Deployment (Flask)](#local-deployment-flask)
7. [Remote Deployment (WebRTC)](#remote-deployment-webrtc)
8. [Additional Implementation Details](#additional-implementation-details)
9. [Future Work](#future-work)
10. [References](#references)

---

## Overview

Traffic signs play a critical role in road safety and infrastructure. With the rise of computer vision applications and deep learning methods, automatically detecting and classifying traffic signs is increasingly important for various fields—ranging from autonomous driving systems to smart city solutions.

This project demonstrates a real-time traffic sign recognition system that:
- Detects and classifies traffic signs in real time (camera feed).
- Uses **YOLOv5** (single-stage object detector) for robust performance.
- Can operate both locally (via Flask + OpenCV) or remotely (via a WebRTC server).
- Provides optional text-to-speech alerts to announce detected signs.

---

## Dataset

- **Source:** [Fırat University Big Data and AI Laboratory](http://buyukveri.firat.edu.tr/veri-setleri/)
- **Details:**
  - 1250 images of traffic signs.
  - 41 distinct classes (commonly used in Turkey).
  - Images include various angles, weather conditions, and lighting scenarios.
  - Originally annotated in Pascal VOC (`.xml` files).

The dataset is split into:
- **Train set:** 80% of images
- **Validation (Test) set:** 20% of images

This split helps the model generalize effectively and prevents overfitting.

---

## Annotation Conversion

The original dataset annotations were in **Pascal VOC** format. YOLO, however, requires a different label format (`.txt` files containing class ID and normalized bounding box coordinates).

A custom converter script was used to:
1. Parse the `.xml` annotations (filename, width, height, object names, bounding box coordinates).
2. Convert these to YOLO-format text files, each line containing:
class_id x_center y_center width height
where all coordinates are normalized to the image dimensions.
3. Create/update the `data.yaml` file (for YOLOv5) specifying:
- Number of classes (`nc = 41`)
- Class names (mapped from 0 to 40)

---

## Model Training

### Why YOLOv5?
- **Single-stage** object detection approach with a balance of speed and accuracy.
- Multiple model sizes available: `yolov5s`, `yolov5m`, `yolov5l`, `yolov5x`.
- Easy to integrate with PyTorch.

### Training Configurations
1. **Model Variants:**  
- `yolov5s` (fewer parameters, faster inference)  
- `yolov5m` (more parameters, potentially higher accuracy)

2. **Input Image Sizes:**  
- `640x640` (faster processing)  
- `1280x1280` (better accuracy but slower)

3. **Hyperparameters:**  
- Epochs: 50 (typical initial choice)
- Batch sizes: 8 or 16 (depending on GPU memory)
- Learning rate: default YOLOv5 settings, adjusted slightly per experiment

### Training Environment
- GPU-enabled environment (Linux, CUDA-supported).
- Model weights and logs can be easily transferred to Windows for testing.

---

## Training Results

During training, YOLOv5 logs several metrics:
- **`train/box_loss`, `train/obj_loss`, `train/cls_loss`**: Training losses for bounding boxes, objectness, and classification.
- **`val/box_loss`, `val/obj_loss`, `val/cls_loss`**: Validation losses on held-out data.
- **`Precision`, `Recall`, `mAP@0.5`, `mAP@0.5:0.95`**: Key indicators of detection and classification performance.

### Observations
- The losses typically start high and decrease significantly as epochs increase.
- Overfitting was not significant, indicated by similar trends in training and validation losses.
- **mAP** and **Precision/Recall** scores improved noticeably around epoch 20-30, then plateaued, suggesting the model reached a performance “saturation.”

Larger image sizes (e.g., `1280x1280`) and bigger models (`yolov5m`) yielded higher **mAP** but slower inference. On the other hand, `yolov5s` with `640x640` input size delivered higher FPS but slightly lower accuracy.

---

## Local Deployment (Flask)

A simple **Flask** web application was created to:
1. Capture a real-time video feed from the local camera (via OpenCV).
2. Process each frame using the trained YOLOv5 model.
3. Display bounding boxes and class names directly on the video feed.
4. Optionally save cropped images of detected signs, rendering them below the video preview.
5. Offer text-to-speech (TTS) alerts using the `pyttsx3` library (e.g., “Stop,” “Yield,” “Speed Limit 30,” etc.).

**Key Steps for Local Use:**
1. Install dependencies.
2. Run the Flask app.
3. Access the local URL (http://127.0.0.1:5000/) in a web browser.
4. Allow camera access and start detection.

---

## Remote Deployment (WebRTC)

A second approach allows real-time detection on a **remote server** (e.g., a Windows 11 VM on Proxmox).

1. **Server-side** uses:
- [`aiortc`](https://aiortc.readthedocs.io/en/latest/) for WebRTC connections.
- YOLOv5 to process incoming video frames.
- A return video stream, annotated with bounding boxes, is sent back to the client.

2. **Client (Browser) side**:
- `getUserMedia()` captures the local webcam feed.
- A WebRTC `RTCPeerConnection` sends the video stream to the server.
- Receives the processed (bounding box) video stream and displays it.
- Optionally uses browser-based TTS (`speechSynthesis`) to announce detected signs.

3. **Secure Tunnel**:
- `WireGuard` can be used to establish a secure, encrypted connection to the remote server, allowing near-local network performance.

### Notes on Browser Security
- Some browsers enforce HTTPS for camera access. During testing, you may need to allow “insecure origins” in Chrome (e.g., `chrome.exe --unsafely-treat-insecure-origin-as-secure=http://YOUR_IP:PORT`).
- For production, always use TLS/SSL certificates.

---

## Additional Implementation Details

- **Cross-platform Path Handling**:  
On Windows, direct loading of YOLOv5 models trained on Linux can lead to path inconsistencies. Using `pathlib` or setting `PosixPath = WindowsPath` can mitigate path-related errors.

- **Performance Tuning**:  
- `yolov5s` + `640x640`: Higher FPS (20-30+ FPS), moderate accuracy.
- `yolov5m` + `1280x1280`: Better detection accuracy, lower FPS.
- Possible future optimizations: [TensorRT](https://github.com/NVIDIA/TensorRT), [OpenVINO](https://github.com/openvinotoolkit/openvino).

- **TTS Integration**:
- **Local Mode**: Python-based `pyttsx3`.
- **Remote Mode**: Browser-based `speechSynthesis`.

---

## Future Work
- **Model Optimization**: Experiment with different backbones (EfficientNet, MobileNet) or accelerate using TensorRT/OpenVINO for edge devices.
- **Mobile Application**: Integrate into iOS/Android with on-device ML or server-side inference.

---

## References
- [Fırat University Big Data and AI Laboratory](http://buyukveri.firat.edu.tr/veri-setleri/)
- [YOLOv5 (Ultralytics)](https://github.com/ultralytics/yolov5)
- [OpenCV](https://opencv.org/)
- [PyTorch](https://pytorch.org/)
- [WebRTC](https://webrtc.org/)
- [aiortc](https://aiortc.readthedocs.io/en/latest/)
- [Flask](https://palletsprojects.com/p/flask/)
- [pyttsx3](https://pypi.org/project/pyttsx3/)

---
