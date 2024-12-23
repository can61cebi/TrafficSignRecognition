import torch
import pathlib
import cv2
import pyttsx3
import time
import os
from flask import Flask, render_template, Response, jsonify, request
from multiprocessing import Process, Queue

# PosixPath'i WindowsPath ile eşleştir (Windows ortamında sorun yaşamamak için).
pathlib.PosixPath = pathlib.WindowsPath

app = Flask(__name__)

# "static/crops" klasörünü oluştur (kırpılmış resimler için)
os.makedirs("static/crops", exist_ok=True)

# Varsayılan kamera indeksi
camera_index = 0
# Global kamera nesnesi
camera = cv2.VideoCapture(camera_index)
camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

# YoloV5 modelini yüklüyoruz (kendi yolunuzu ayarlayın)
model = torch.hub.load(
    'yolov5_model',
    'custom',
    path='yolov5_model/weights/best.pt',
    source='local',
    force_reload=True
)

# Konsola yazım aralığı takibi
last_console_print_time = 0

# Ses kuyruğu
label_queue = Queue()

# Tespit edilen veriyi tutan sözlük
DETECTION_DATA = {
    "detected": False,
    "signs": []
}


def find_cameras(max_tested=5):
    """
    0'dan max_tested'e kadar indisleri deneyerek açılabilen kameraları döndürür.
    """
    available = []
    for i in range(max_tested):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            available.append(i)
            cap.release()
    return available


def speech_process(label_queue):
    """
    Ayrı bir süreçte çalışacak TTS fonksiyonu.
    """
    engine = pyttsx3.init()
    engine.setProperty('voice', 'tr')  # Türkçe ses
    last_speech_time = time.time()
    detected_labels = set()

    while True:
        try:
            while True:
                label = label_queue.get(timeout=0.1)
                detected_labels.add(label)
        except:
            pass

        current_time = time.time()
        # 6 saniyede bir konuşma
        if current_time - last_speech_time >= 6:
            if detected_labels:
                label_to_speak = next(iter(detected_labels))
                engine.say(label_to_speak)
                engine.runAndWait()
                detected_labels.clear()
            last_speech_time = current_time

        time.sleep(0.1)


def generate_frames():
    """
    OpenCV ile kareleri okuyup MJPEG formatında tarayıcıya akıtan fonksiyon.
    """
    global last_console_print_time, DETECTION_DATA
    while True:
        success, frame = camera.read()
        if not success:
            print("Kameradan görüntü alınamıyor.")
            break
        else:
            # Yolo ile tahmin
            results = model(frame)
            labels, cords = results.xyxyn[0][:, -1], results.xyxyn[0][:, :-1]

            # Bu kare için tespitler
            current_detections = []
            current_time = time.time()
            y_offset = 50

            for label, cord in zip(labels, cords):
                x1, y1, x2, y2, conf = cord
                if conf > 0.3:  # Güven skoru eşiği
                    x1 = int(x1 * frame.shape[1])
                    y1 = int(y1 * frame.shape[0])
                    x2 = int(x2 * frame.shape[1])
                    y2 = int(y2 * frame.shape[0])

                    label_name = results.names[int(label)]

                    # Kare üzerine dikdörtgen & etiket
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(
                        frame, label_name, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2
                    )

                    # Kuyruğa ekle (seslendirilecek)
                    label_queue.put(label_name)

                    # Konsola yazdırma (1 saniyede bir)
                    if current_time - last_console_print_time >= 1:
                        print(f"Tespit edilen nesne: {label_name}")
                        last_console_print_time = current_time

                    # Levha görüntüsünü kırpıp kaydet
                    # time.time_ns() ile unique bir ismi çok büyük ihtimalle çakışmasız şekilde üretiyoruz
                    crop_filename = f"crop_{time.time_ns()}.jpg"
                    crop_path = os.path.join("static", "crops", crop_filename)
                    cropped = frame[y1:y2, x1:x2]
                    cv2.imwrite(crop_path, cropped)

                    # Bu karede tespit edilenleri listeye ekle
                    current_detections.append({
                        "name": label_name,
                        "img_url": f"/static/crops/{crop_filename}"
                    })

                    # Video üzerine de "Tespit" yazısı
                    cv2.putText(
                        frame,
                        f"Tespit: {label_name}",
                        (10, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (0, 0, 255),
                        2
                    )
                    y_offset += 30

            # DETECTION_DATA'yı güncelle
            if current_detections:
                DETECTION_DATA["detected"] = True
                DETECTION_DATA["signs"] = current_detections
            else:
                DETECTION_DATA["detected"] = False
                DETECTION_DATA["signs"] = []

            # Kareyi JPEG formatına çevirip gönder
            ret, buffer = cv2.imencode('.jpg', frame)
            if ret:
                frame_bytes = buffer.tobytes()
                yield (
                    b'--frame\r\n'
                    b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n'
                )


@app.route('/')
def index():
    """
    Ana sayfa (index.html).
    """
    return render_template('index.html')


@app.route('/video_feed')
def video_feed():
    """
    <img> etiketinin src ile bağlandığı MJPEG endpoint.
    """
    return Response(
        generate_frames(),
        mimetype='multipart/x-mixed-replace; boundary=frame'
    )


@app.route('/get_cameras', methods=['GET'])
def get_cameras():
    """
    Sistemden açılabilen kamera indekslerini JSON olarak döndürür.
    Örnek: [0, 1, 2]
    """
    cams = find_cameras(max_tested=5)
    return jsonify(cams)


@app.route('/set_camera', methods=['POST'])
def set_camera():
    """
    { "camera_index": X } verisi alır,
    global camera değişkenini yeni indekse çevirir.
    """
    global camera, camera_index

    data = request.json
    new_index = data.get('camera_index', 0)

    # Eski kamerayı kapat
    camera.release()

    # Yeni kamerayı aç
    camera_index = new_index
    camera = cv2.VideoCapture(camera_index)
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

    return jsonify({"status": "ok", "current_camera": camera_index})


@app.route('/get_detection_info')
def get_detection_info():
    """
    Tespit edilen levhaların bilgisini JSON olarak döndürür.
    {
      "detected": bool,
      "signs": [
        {"name": "xxx", "img_url": "/static/crops/xxx.jpg"},
        ...
      ]
    }
    """
    return jsonify(DETECTION_DATA)


if __name__ == "__main__":
    # Ses sürecini başlat
    p = Process(target=speech_process, args=(label_queue,))
    p.start()

    try:
        print("Flask sunucusu başlıyor... http://127.0.0.1:5000")
        app.run(host='0.0.0.0', port=5000, debug=False)
    finally:
        p.terminate()
        p.join()
