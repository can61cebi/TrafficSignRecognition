# Mühendislikte Disiplinlerarası Proje Dersi

### 1. **Proje Yapısının Hazırlanması:**
   - Flask uygulaması için gerekli dosya yapısını oluştur.
   - Gerekli kütüphaneleri yükle (Flask, OpenCV, YoloV5, pyttsx3, vs).

### 2. **Veri Setinin İncelenmesi:**
   - Verileri anlamak ve veri ön işleme adımlarını hazırlamak. Test ve train verilerinin nasıl işleneceğini inceleyeceğiz.

### 3. **YoloV5 Modelinin Eğitilmesi ya da Hazır Model Kullanımı:**
   - YoloV5 ile trafik işaretlerini tanıyacak bir model kullanacağız. Burada mevcut hazır bir model kullanabiliriz ya da eğitim aşamasını gerçekleştirebiliriz.

### 4. **OpenCV ile Kamera Entegrasyonu:**
   - Kullanıcının web kamerası ile bağlantı kurmak ve video akışını işlemek.

### 5. **Trafik İşareti Tanıma:**
   - OpenCV ve YoloV5 ile kamera görüntülerinden trafik işaretlerini tespit etme.

### 6. **Sonuçların Görüntülenmesi ve Sesli Çıktı:**
   - Tespit edilen trafik işaretini ekranda gösterme ve pyttsx3 ile Türkçe olarak sesli okuma.

### 7. **Flask ile Web Arayüzü Oluşturma:**
   - Web tabanlı bir arayüz hazırlayacağız, burada kamera akışını gösterecek ve sonuçları sunacağız.

Bu adımlara başladıkça her birini detaylandıracağız. Şimdi ilk adıma, **Proje Yapısının Hazırlanmasına** geçelim.

### **Proje Yapısının Hazırlanması**

**Gerekli Kütüphaneler:**
   - Flask: Web framework.
   - OpenCV: Görüntü işleme.
   - Pyttsx3: Sesli çıktı.
   - YoloV5: Trafik işaretlerini tanımak için model.

İlk olarak, bu kütüphaneleri yükleyelim. Aşağıdaki komutları terminal üzerinden çalıştırabilirsin:

```bash
pip install Flask opencv-python pyttsx3 torch torchvision
```


### 1. **Gerekli Klasör ve Dosyaların Yapılandırılması:**

Proje yapısında birkaç ek dosya oluşturacağız:

1. `yolov5_model/` — Bu klasörde YoloV5 model dosyalarını tutacağız.
2. `static/` ve `templates/` klasörlerini oluşturalım:
   - `static/css/`: CSS dosyaları için.
   - `static/js/`: JavaScript dosyaları için.
   - `templates/`: Flask'ın kullanacağı HTML dosyalarını buraya ekleyeceğiz (örneğin `index.html`).

Bunları projenin kök dizinine ekledikten sonra Flask ile basit bir web arayüzü oluşturmaya başlayabiliriz.

### 2. **Flask ile Temel Web Uygulamasını Kurma:**

İlk olarak, basit bir Flask uygulaması oluşturalım. `main.py` dosyanda temel bir Flask yapılandırması kuracağız.

Aşağıdaki kodu `main.py` dosyana ekle:

```python
from flask import Flask, render_template, Response
import cv2

app = Flask(__name__)

# OpenCV ile kamerayı başlatıyoruz
camera = cv2.VideoCapture(0)

def generate_frames():
    while True:
        success, frame = camera.read()  # Kameradan görüntü al
        if not success:
            break
        else:
            ret, buffer = cv2.imencode('.jpg', frame)
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
    app.run(debug=True)
```

Bu kod, Flask uygulamasını başlatır ve kamera akışını web arayüzüne yönlendirir. Ayrıca `index.html` sayfasında kamera görüntüsünü göstereceğiz.

### 3. **HTML Sayfasının Oluşturulması:**

Şimdi `templates/index.html` dosyasını oluştur ve aşağıdaki kodu ekle:

```html
<!DOCTYPE html>
<html lang="tr">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Trafik İşareti Tanıma</title>
</head>
<body>
    <h1>Canlı Kamera Akışı</h1>
    <img src="{{ url_for('video_feed') }}" width="600">
</body>
</html>
```

Bu sayfa, Flask ile oluşturulan kamera akışını gösterecek.

### 4. **Proje İçinde YoloV5 Modelinin Kurulumu:**

YoloV5 modelini kullanacağız. Burada iki seçeneğimiz var: Kendi modelimizi eğitebiliriz veya önceden eğitilmiş bir YoloV5 modelini yükleyip kullanabiliriz.

#### YoloV5 Modelinin Yüklenmesi:
GitHub üzerinden YoloV5'i projenize klonlayın:

```bash
git clone https://github.com/ultralytics/yolov5 yolov5_model/
```

`yolov5_model/` dizini içinde YoloV5 model dosyalarınızı bulunduracağız. YoloV5’i PyTorch ile kullanarak trafik işaretlerini tespit edeceğiz.

Harika, kamerayı çalıştırmayı başardığına sevindim! Şimdi YoloV5 ile trafik işaretlerini tanıma aşamasına geçebiliriz. Bu aşamada, YoloV5 modelini kullanarak trafik işaretlerini tespit edeceğiz ve bu tespitleri kullanarak modelimizi Flask uygulamasına entegre edeceğiz.

### YOLOv5 Entegrasyonu Adımları

1. **YOLOv5 Kurulumu:**
   YoloV5'i projenin kök dizinine klonlayarak başlayalım. Daha önce belirttiğimiz gibi, bunu yapabilirsin:

   ```bash
   git clone https://github.com/ultralytics/yolov5 yolov5_model/
   ```
   
   Bu komut, YoloV5'i `yolov5_model/` klasörüne klonlayacak.

2. **Model Ağırlıklarının Yüklenmesi:**
   Önceden eğitilmiş YoloV5 model ağırlıklarını indirmemiz gerekiyor. YoloV5, çeşitli önceden eğitilmiş ağırlık dosyalarını sunar (örneğin, `yolov5s.pt`). Ağırlıkları `yolov5_model/weights/` klasörüne indirmen gerekiyor:

   ```bash
   # yolov5_model dizininde olmalısın
   cd yolov5_model
   mkdir weights
   wget https://github.com/ultralytics/yolov5/releases/download/v6.0/yolov5s.pt -P weights/
   ```

   Bu ağırlık dosyasını kullanarak trafik işaretlerini tespit edeceğiz.

3. **Modeli Python ile Kullanmak:**
   `main.py` dosyanda YoloV5 modelini kullanarak gelen görüntüyü işlemek için bir işlev ekleyeceğiz. Aşağıda, bir çerçeveyi işlemek için YoloV5 kullanımı örnek olarak gösterilmiştir:

   ```python
   sadece
   import torch
   from flask import Flask, render_template, Response
   import cv2
   import pyttsx3
   import time
   import threading

   app = Flask(__name__)

   camera = cv2.VideoCapture(1)

   model = torch.hub.load('yolov5_model', 'custom', path='yolov5_model/weights/yolov5s.pt', source='local')

   engine = pyttsx3.init()
   engine.setProperty('voice', 'tr')
   engine_lock = threading.Lock()

   def say_label(label_name):
    with engine_lock:
        engine.say(label_name)
        engine.runAndWait()


   def generate_frames():
    while True:
        success, frame = camera.read()
        if not success:
            print("Kameradan görüntü alınamıyor.")
            break
        else:
            results = model(frame)

            labels, cords = results.xyxyn[0][:, -1], results.xyxyn[0][:, :-1]
            for label, cord in zip(labels, cords):
                x1, y1, x2, y2, conf = cord
                if conf > 0.3:  # Güven skoruna göre filtreleme (0.5 yerine daha düşük bir değer kullanarak)
                    x1, y1, x2, y2 = int(x1 * frame.shape[1]), int(y1 * frame.shape[0]), int(x2 * frame.shape[1]), int(y2 * frame.shape[0])
                    label_name = results.names[int(label)]

                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, label_name, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)

                    threading.Thread(target=say_label, args=(label_name,)).start()

                    print(f"Tespit edilen nesne: {label_name}")
                    cv2.putText(frame, f"Tespit: {label_name}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

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
    print("Flask sunucusu başlatılıyor... Tarayıcınızda http://127.0.0.1:5000 adresine gidin.")
    app.run(host='0.0.0.0', port=5000, debug=True)
   ```

   Açıklamalar:
   - **YOLOv5 Modelini Yüklemek:** `torch.hub.load()` ile `yolov5_model` klasöründen modeli yüklüyoruz. Bu, lokal dosyaları kullanarak YoloV5'i entegre eder.
   - **Modeli Kullanarak Tespit Yapmak:** `results = model(frame)` ile kameradan alınan görüntü üzerinde tespit işlemi yapıyoruz.
   - **Güven Skoru ve Kutucuk Çizme:** Tespit edilen her nesne için güven skoru `conf` ile filtre yaparak, sadece belirli bir güven skoru üzerindeki nesneleri görüntüde gösteriyoruz.
   - **Pyttsx3 ile Sesli Çıktı:** Tespit edilen nesnenin adını `engine.say()` kullanarak Türkçe olarak seslendiriyoruz.

Bu aşamalar tamamlandıktan sonra modeli trafik işaretlerini tanıması yönünde eğiteceğiz.

Trafik işaretlerini tanımak üzere YoloV5 modelini kendi veri setinizle eğitmek için belirli adımlar atmanız gerekiyor. YoloV5’i, özel olarak trafik işaretlerini tanımak üzere eğitmek için aşağıdaki adımları takip edelim:

### 1. **Veri Setinizi Hazırlama**
Öncelikle veri setinizin Yolo formatında olduğundan emin olmalısınız. Yolo formatında her bir görüntü dosyası için ona karşılık gelen `.txt` dosyaları olmalıdır. Bu `.txt` dosyalarında her satır, bir nesnenin sınıf etiketi ve normalleştirilmiş koordinatlarını içerir.

#### Yolo Veri Seti Formatı:
- Her görüntünün `.jpg` veya `.png` dosyası olacaktır.
- Aynı isimle her görüntü için bir `.txt` dosyası olacak.
- `.txt` dosyası formatı:
  ```
  sınıf_id x_center y_center width height
  ```
  - `sınıf_id`: Sınıf etiketi (0, 1, 2... gibi).
  - `x_center`, `y_center`, `width`, `height`: Normalleştirilmiş koordinatlar (0 ile 1 arasında).

Örneğin, bir "Dur" işareti için `.txt` dosyanız şu şekilde olabilir:
```
0 0.5 0.5 0.3 0.4
```

#### Veri Seti Yapısı:
Aşağıdaki gibi bir dizin yapısına sahip olmalısınız:
```
/traffic_sign_dataset
    /images
        /train
            image1.jpg
            image2.jpg
            ...
        /val
            imageA.jpg
            imageB.jpg
            ...
    /labels
        /train
            image1.txt
            image2.txt
            ...
        /val
            imageA.txt
            imageB.txt
            ...
```

### 2. **`data.yaml` Dosyasını Oluşturma**
YoloV5 eğitimi için bir konfigürasyon dosyasına ihtiyacımız var. Bu dosya, veri setinizin yolunu, sınıf sayısını ve sınıf adlarını içerir. Bu dosyayı `traffic_sign_dataset/data.yaml` olarak oluşturabilirsiniz:

```yaml
train: ../traffic_sign_dataset/images/train
val: ../traffic_sign_dataset/images/val

nc: 3  # Sınıf sayısı (örneğin, "Dur", "Yol Ver", "Hız Limiti")
names: ['Dur', 'Yol Ver', 'Hız Limiti']  # Sınıf isimleri
```

Burada:
- `train`: Eğitim veri setinin yolu.
- `val`: Doğrulama veri setinin yolu.
- `nc`: Sınıf sayısı.
- `names`: Sınıf isimleri.

### 3. **YoloV5'i Eğitmek İçin Ortamı Hazırlama**
Eğitim işlemini gerçekleştirmek için YoloV5 ortamını hazırlamanız gerekiyor. Bunun için:

- YoloV5 dizininde olduğunuzdan emin olun:
  ```bash
  cd yolov5_model
  ```
- Gereksinimleri yükleyin:
  ```bash
  pip install -r requirements.txt
  ```

### 4. **Modeli Eğitme**
YoloV5'i trafik işaretlerinizi tanıyacak şekilde eğitmek için `train.py` scriptini çalıştırmalısınız:

```bash
python train.py --img 640 --batch 16 --epochs 50 --data traffic_sign_dataset/data.yaml --weights yolov5s.pt --cache
```

Açıklamalar:
- `--img 640`: Girdi görüntüsünün boyutu. 640, Yolo için yaygın bir boyuttur ve performans ile hız arasında iyi bir denge sağlar.
- `--batch 16`: Her iterasyonda kaç görüntü ile eğitim yapılacağını belirtir.
- `--epochs 50`: Eğitim için toplam dönem sayısı.
- `--data traffic_sign_dataset/data.yaml`: Eğitim için kullanılan veri seti.
- `--weights yolov5s.pt`: Eğitimin başladığı önceden eğitilmiş model. YoloV5s, daha küçük ve hızlı bir modeldir.
- `--cache`: Veri setini bellekte tutarak daha hızlı eğitim yapılmasını sağlar.

### 5. **Eğitim Sonrası Ağırlıkların Kullanımı**
Eğitim tamamlandığında, sonuçlar `runs/train/exp` klasörüne kaydedilecektir. Burada `best.pt` ve `last.pt` ağırlık dosyaları bulunur. Bu ağırlık dosyalarını kullanarak modeli çalıştırabilirsiniz.

### 6. **Modeli Kullanma (Yeniden Entegre Etme)**
Eğitimden sonra, `main.py` dosyanızda bu yeni ağırlıkları kullanarak modeli entegre edin:

```python
# Eğitilmiş modeli yükle
model = torch.hub.load('yolov5_model', 'custom', path='runs/train/exp/best.pt', source='local')
```

Bu satırda, yeni eğitilmiş modelinizi kullanarak trafik işaretlerini tanımaya başlayabilirsiniz.

### 7. **Modeli Test Etme**
Modeli Flask tabanlı uygulamanızda test edebilir ve doğru şekilde trafik işaretlerini tanıdığından emin olabilirsiniz. Modelin performansını artırmak için daha fazla epoch kullanabilir veya batch boyutunu artırabilirsiniz.

### Ek Notlar:
- **Görüntü Kalitesi:** Eğitim sırasında görüntülerin kalitesi önemlidir. Düşük kaliteli görüntüler eğitim sonucunu olumsuz etkileyebilir.
- **Veri Sayısı:** Daha fazla eğitim verisi genellikle daha iyi sonuç verir. Veri setinizde çeşitli açılardan ve farklı aydınlatma koşullarında çekilmiş çok sayıda görüntü bulundurmanız modelin doğruluğunu artıracaktır.
- **Augmentasyon (Veri Çoğaltma):** Eğitim sırasında veri augmentasyonu (döndürme, kırpma, parlaklık artırma vs.) uygulamak modelin genelleme kapasitesini artıracaktır. YoloV5, eğitim sırasında augmentasyonu otomatik olarak uygulayabilir.

