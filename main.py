import torch
from flask import Flask, render_template, Response
import cv2
import pyttsx3
import time
import threading

# Flask uygulaması
app = Flask(__name__)

# OpenCV ile kamerayı başlatıyoruz
camera = cv2.VideoCapture(1)

# YoloV5 modelini yükle
model = torch.hub.load('yolov5_model', 'custom', path='yolov5_model/weights/yolov5s.pt', source='local')

# Pyttsx3 motoru kur
engine = pyttsx3.init()
engine.setProperty('voice', 'tr')  # Türkçe dil desteği
engine_lock = threading.Lock()  # Pyttsx3 motorunu senkronize etmek için kilit

def say_label(label_name):
    with engine_lock:
        engine.say(label_name)
        engine.runAndWait()


def generate_frames():
    while True:
        # Kameradan görüntü al
        success, frame = camera.read()
        if not success:
            print("Kameradan görüntü alınamıyor.")
            break
        else:
            # YoloV5 ile görüntüyü işleyelim
            results = model(frame)

            # Tespit edilen sonuçları alalım ve işleyelim
            labels, cords = results.xyxyn[0][:, -1], results.xyxyn[0][:, :-1]
            for label, cord in zip(labels, cords):
                x1, y1, x2, y2, conf = cord
                if conf > 0.3:  # Güven skoruna göre filtreleme (0.5 yerine daha düşük bir değer kullanarak)
                    x1, y1, x2, y2 = int(x1 * frame.shape[1]), int(y1 * frame.shape[0]), int(x2 * frame.shape[1]), int(y2 * frame.shape[0])
                    label_name = results.names[int(label)]

                    # Kutucuk çiz ve etiketi yaz
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, label_name, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)

                    # Türkçe olarak etiketi sesli oku (asenkron olarak)
                    threading.Thread(target=say_label, args=(label_name,)).start()

                    # Etiketi konsola ve web sayfasına yazdır
                    print(f"Tespit edilen nesne: {label_name}")
                    cv2.putText(frame, f"Tespit: {label_name}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            # Görüntüyü tarayıcıya gönder
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
    # Flask sunucusunu başlatma
    print("Flask sunucusu başlatılıyor... Tarayıcınızda http://127.0.0.1:5000 adresine gidin.")
    app.run(host='0.0.0.0', port=5000, debug=True)