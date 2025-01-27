# Real-Time Traffic Sign Recognition with YOLOv5

This repository demonstrates a real-time traffic sign detection and classification system using **YOLOv5**. The project includes:

- A **Flask-based** local deployment (with OpenCV for camera capture and optional text-to-speech).
- A **WebRTC-based** remote deployment (using `aiortc`) for serving bounding-box annotated video streams from a server to browser clients.
- A dataset of 41 traffic sign classes (from [Fırat University Big Data and AI Laboratory](http://buyukveri.firat.edu.tr/veri-setleri/)).
- Code for annotation format conversion (Pascal VOC to YOLO).
- Various training experiments with different YOLOv5 model sizes and image resolutions.

Below is an outline of the repository’s features and usage.

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

## 1. Overview

Traffic signs are crucial for road safety and infrastructure management. As deep learning and computer vision tools have become more accessible, automated detection and classification of these signs has gained importance across various industries (e.g., **autonomous vehicles**, **smart city planning**).

This project implements a **real-time traffic sign recognition system** that:
- Uses **YOLOv5** for object detection and classification of traffic signs.
- Supports both **local** (Flask) and **remote** (WebRTC) deployments.
- Provides **text-to-speech (TTS)** alerts announcing the detected signs.
- Optionally saves cropped images of the detected signs.
- Demonstrates how to serve the processed video back to end-users via a browser or local interface.

<p align="center">
  <img src="https://github.com/can61cebi/TrafficSignRecognition/blob/master/presentation/sonuc1.png" alt="webpage ui v1" width="500">
</p>
<p align="center"><em>Figure 1. Example local web interface using Flask</em></p>

---

## 2. Dataset

- **Source:** [Fırat University Big Data and AI Laboratory](http://buyukveri.firat.edu.tr/veri-setleri/)
- **Images:** 1250 total, capturing various angles, lighting conditions, and weather scenarios.
- **Classes:** 41 traffic sign types common in Turkey.
- **Annotations:** Provided in **Pascal VOC** format (XML).

### Train/Test Split

- **Training Set:** ~80% of images
- **Validation/Testing Set:** ~20% of images

This split ensures sufficient coverage of different classes in both training and evaluation stages.

---

## 3. Annotation Conversion

YOLOv5 requires annotation files in a text-based format (`.txt`) with class IDs and **normalized** bounding box coordinates. Because the dataset annotations were originally in **Pascal VOC**, a conversion script was utilized to:

1. Read each `.xml` file (bounding boxes, image size, class names).
2. Convert bounding box coordinates into YOLO format:  
   `class_id x_center y_center width height`  
   (all normalized by the image’s width and height).
3. Generate `.txt` labels in the appropriate directories corresponding to each image.
4. Update or create a `data.yaml` file for YOLOv5 with:
   - Number of classes: `nc = 41`
   - List of class names, indexed from 0 to 40.

---

## 4. Model Training

### Why YOLOv5?

- **Single-stage object detection** with an excellent balance between speed and accuracy.
- Easily available PyTorch implementation.
- Multiple model sizes (`yolov5s`, `yolov5m`, `yolov5l`, `yolov5x`) to accommodate different performance requirements.

### Training Configurations

1. **Model Variants**  
   - `yolov5s`: Fewer parameters, faster inference.  
   - `yolov5m`: More parameters, potentially higher accuracy.

2. **Input Resolutions**  
   - `640x640`: Faster processing.  
   - `1280x1280`: Better accuracy but increased GPU usage and lower FPS.

3. **Hyperparameters**  
   - Epochs: 50  
   - Batch Size: 8 or 16 (depending on available GPU memory)  
   - Learning Rate: Typically uses YOLOv5 defaults, with slight adjustments if needed.

### Environment

- Trained in a **Linux** environment with **CUDA** support (GPU).
- Weights can be transferred cross-platform (e.g., to Windows 11 for inference).

---

## 5. Training Results

Throughout training, YOLOv5 logs the following metrics:

- **Training Losses:** `train/box_loss`, `train/obj_loss`, `train/cls_loss`  
- **Validation Losses:** `val/box_loss`, `val/obj_loss`, `val/cls_loss`  
- **Performance Measures:** `Precision`, `Recall`, `mAP@0.5`, `mAP@0.5:0.95`

**mAP** and **Precision/Recall** typically improved rapidly from epochs 1–30, then plateaued.

<p align="center">
  <img src="https://github.com/can61cebi/TrafficSignRecognition/blob/master/presentation/train_batch0.jpg" alt="Train Batch 0 Results" width="500">
</p>
<p align="center"><em>Figure 2. Sample training batch visualization</em></p>

<p align="center">
  <img src="https://github.com/can61cebi/TrafficSignRecognition/blob/master/presentation/val_batch1_pred.jpg" alt="Validation Batch 1 Results" width="500">
</p>
<p align="center"><em>Figure 3. Sample validation batch with predicted boxes</em></p>

- **`yolov5s + 640x640`**: Achieves higher FPS (~20–30+ on standard GPUs) with a slight drop in accuracy.
- **`yolov5m + 1280x1280`**: Improves accuracy but lowers real-time performance.

---

## 6. Local Deployment (Flask)

A **Flask** application (`main.py`) provides a simple interface to capture video frames from a local camera, perform object detection using YOLOv5, and render bounding boxes on the live feed. 

### Key Features

1. **Camera Capture (OpenCV)**  
   - Configurable camera index and resolution.
   - Automatic detection of available cameras (up to a specified index).

2. **Real-time Inference**  
   - Frames are passed to the YOLOv5 model for predictions.
   - Classes with confidence above a threshold (e.g., 0.3) are annotated.

3. **Cropped Images**  
   - When a sign is detected, the region of interest is cropped and saved to `static/crops/`.
   - These cropped images are displayed below the detection stream.

4. **Text-to-Speech (TTS)**  
   - A separate process (via `multiprocessing.Process`) handles TTS to avoid blocking.
   - `pyttsx3` is used for local speech synthesis (in Turkish in the provided sample).

5. **Bootstrap-based UI**  
   - A responsive web page (`templates/index.html`) using Bootstrap 5.
   - Automatically refreshes detection info via periodic AJAX calls.

Below is a simplified flow of **`main.py`**:

```python
# Load YOLOv5 model
model = torch.hub.load(
    'yolov5_model',
    'custom',
    path='yolov5_model/weights/best.pt',
    source='local',
    force_reload=True
)

# Start a TTS process
p = Process(target=speech_process, args=(label_queue,))
p.start()

# ... Set up Flask routes, camera feed, etc.

# Run the Flask server
app.run(host='0.0.0.0', port=5000, debug=False)
```

**Usage:**
1. Install required libraries (`requirements.txt` or manual installation of `Flask`, `torch`, `opencv-python`, `pyttsx3`, etc.).
2. Run `python main.py`.
3. Go to [http://127.0.0.1:5000/](http://127.0.0.1:5000/) to view the live detection feed.

<p align="center">
  <img src="https://github.com/can61cebi/TrafficSignRecognition/blob/master/presentation/sonuc3.jpg" alt="webpage ui v2 with cropped images" width="500">
</p>
<p align="center"><em>Figure 4. Local Flask interface with cropped sign images displayed</em></p>

---

## 7. Remote Deployment (WebRTC)

For remote usage, the repository provides `server.py` which uses **WebRTC** to transfer live video from the client’s browser to the server and sends back annotated frames in real-time.

### How It Works

1. **Client (Browser)**
   - An HTML file (`index.html`) acquires media from the user's webcam via `getUserMedia()`.
   - It sends the video stream to the server using a **WebRTC** `RTCPeerConnection`.
   - Receives the processed (detected bounding boxes) stream from the server.
   - Optionally announces detected signs using the browser’s built-in TTS (`speechSynthesis`).

2. **Server (Python)**
   - Receives the video track.
   - Uses the **YOLOv5** model to run inference on each incoming frame.
   - Overlays bounding boxes and label text on the frame.
   - Sends the annotated frames back to the client over the same WebRTC connection.
   - Communicates detection metadata to the client via a **DataChannel** in JSON format.

3. **Setup**  
   - Install dependencies like `aiortc` (`pip install aiortc`), `aiohttp`, `torch`, and `opencv-python`.
   - Run `python server.py`.
   - Access the server’s [index page](http://YOUR_SERVER_IP:8080/) from a WebRTC-capable browser (Chrome, Firefox, etc.).

<p align="center">
  <img src="https://github.com/can61cebi/TrafficSignRecognition/blob/master/presentation/sonuc4.jpg" alt="webpage ui v3 webrtc remote server version" width="500">
</p>
<p align="center"><em>Figure 5. Remote WebRTC interface with bounding boxes rendered server-side</em></p>

#### Browser Security Note
- Some browsers enforce HTTPS for camera access. For development, you may launch Chrome with specific flags:
  ```bash
  chrome.exe --unsafely-treat-insecure-origin-as-secure=http://YOUR_IP:PORT
  ```
- For production environments, always use **HTTPS** (TLS) for secure WebRTC connections.

---

## 8. Additional Implementation Details

- **Cross-platform Paths**:  
  In the provided code, `pathlib.PosixPath` is mapped to `pathlib.WindowsPath` to avoid path conflicts across Windows/Linux environments.
  
- **Performance Tuning**:  
  - Smaller models and input sizes yield higher FPS, suitable for real-time edge devices.
  - Larger models with bigger input resolution achieve better detection but may fall below real-time requirements.
  - Further optimizations can be done with [TensorRT](https://github.com/NVIDIA/TensorRT), [OpenVINO](https://github.com/openvinotoolkit/openvino), or by pruning/quantizing the model.

- **Text-to-Speech**:  
  - **Local (Flask)**: Uses Python’s `pyttsx3` library, running in a separate process to avoid blocking detection loops.  
  - **Remote (WebRTC)**: Uses in-browser `speechSynthesis`, freeing the server from TTS tasks and allowing each client to hear detections in their local language settings.

---

## 9. Future Work

1. **Model Optimization**  
   - Experiment with alternative backbones (e.g., EfficientNet, MobileNet).
   - Employ quantization or pruning for edge hardware or mobile devices.

2. **Mobile Application**  
   - Develop a native iOS/Android app that either runs the model locally (on-device ML) or streams video to this server-based solution.

3. **Enhanced UI/UX**  
   - Improve the user interface to highlight or track multiple signs over time.
   - Add logging for historical analytics of detected traffic signs.

---

## 10. References

- [Fırat University Big Data and AI Laboratory](http://buyukveri.firat.edu.tr/veri-setleri/)
- [YOLOv5 (Ultralytics)](https://github.com/ultralytics/yolov5)
- [OpenCV](https://opencv.org/)
- [PyTorch](https://pytorch.org/)
- [WebRTC](https://webrtc.org/)
- [aiortc](https://aiortc.readthedocs.io/en/latest/)
- [Flask](https://palletsprojects.com/p/flask/)
- [pyttsx3](https://pypi.org/project/pyttsx3/)

---

### Thank you for checking out this real-time traffic sign detection project!  
For questions or contributions, please open an issue or create a pull request. Safe driving and happy coding!
