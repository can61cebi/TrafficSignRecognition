import torch
import pathlib  # pathlib modülünü içe aktarıyoruz
from flask import Flask, render_template, Response
import cv2
import pyttsx3
import time
from multiprocessing import Process, Queue

# PosixPath'i WindowsPath ile eşleştiriyoruz
pathlib.PosixPath = pathlib.WindowsPath

# Flask uygulaması
app = Flask(__name__)

# OpenCV ile kamerayı başlatıyoruz
camera = cv2.VideoCapture(1)
camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

# YoloV5 modelini yüklüyoruz
model = torch.hub.load('yolov5_model', 'custom', path='yolov5_model/weights/best.pt', source='local', force_reload=True)

# Zamanlamayı yönetmek için değişkenler
last_console_print_time = 0

# Ses süreciyle iletişim için kuyruk oluşturuyoruz
label_queue = Queue()

def speech_process(label_queue):
    engine = pyttsx3.init()
    engine.setProperty('voice', 'tr')  # Türkçe dil desteği

    last_speech_time = time.time()
    detected_labels = set()
    while True:
        # 3 saniye boyunca etiketleri topluyoruz
        try:
            while True:
                # Bloke etmeyen, zaman aşımı olan get işlemi
                label = label_queue.get(timeout=0.1)
                detected_labels.add(label)
        except:
            # Zaman aşımı gerçekleşti, kuyrukta başka etiket yok
            pass

        current_time = time.time()
        if current_time - last_speech_time >= 6:
            if detected_labels:
                # İlk tespit edilen etiketi okuyoruz
                label_to_speak = next(iter(detected_labels))
                engine.say(label_to_speak)
                engine.runAndWait()
                detected_labels.clear()
            last_speech_time = current_time
        time.sleep(0.1)  # CPU aşırı kullanımını önlemek için küçük bir gecikme

def generate_frames():
    global last_console_print_time
    while True:
        # Kameradan kare alıyoruz
        success, frame = camera.read()
        if not success:
            print("Kameradan görüntü alınamıyor.")
            break
        else:
            # YoloV5 ile kareyi işliyoruz
            results = model(frame)

            # Etiketleri ve koordinatları çıkarıyoruz
            labels, cords = results.xyxyn[0][:, -1], results.xyxyn[0][:, :-1]
            current_time = time.time()
            y_offset = 50  # Kare üzerindeki etiketlerin başlangıç y-koordinatı

            for label, cord in zip(labels, cords):
                x1, y1, x2, y2, conf = cord
                if conf > 0.3:  # Güven skoruna göre filtreleme
                    x1, y1, x2, y2 = int(x1 * frame.shape[1]), int(y1 * frame.shape[0]), int(x2 * frame.shape[1]), int(y2 * frame.shape[0])
                    label_name = results.names[int(label)]

                    # Kareye dikdörtgen ve etiket çiziyoruz
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, label_name, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)

                    # Etiketi ses kuyruğuna ekliyoruz
                    label_queue.put(label_name)

                    # Konsola her 1 saniyede bir yazdırıyoruz
                    if current_time - last_console_print_time >= 1:
                        print(f"Tespit edilen nesne: {label_name}")
                        last_console_print_time = current_time

                    # Web sayfasında etiketleri üst üste binmeden gösteriyoruz
                    cv2.putText(frame, f"Tespit: {label_name}", (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    y_offset += 30  # Bir sonraki etiket için satırı değiştiriyoruz

            # Kareyi tarayıcıya gönderiyoruz
            ret, buffer = cv2.imencode('.jpg', frame)
            if ret:
                frame = buffer.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    # Ses sürecini başlatıyoruz
    p = Process(target=speech_process, args=(label_queue,))
    p.start()

    try:
        # Flask sunucusunu başlatıyoruz
        print("Flask sunucusu başlıyor... Tarayıcınızda http://127.0.0.1:5000 adresine gidin.")
        app.run(host='0.0.0.0', port=5000, debug=False)
    finally:
        # Uygulama kapatıldığında ses sürecini sonlandırıyoruz
        p.terminate()
        p.join()
