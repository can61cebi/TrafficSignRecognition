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

### Önerilen Dizin Yapısı ve Dosya Yerleşimi

#### Önerilen Dosya Yapısı
```
TrafikIsaretlerTanima/
├── .venv/
├── static/
│   ├── css/
│   │   └── style.css
│   └── js/
│       └── script.js
├── templates/
│   └── index.html
├── traffic_signs_dataset_v1/
│   ├── images/
│   │   ├── train/           # Eğitim görüntüleri (JPG)
│   │   └── val/             # Doğrulama görüntüleri (JPG)
│   ├── labels/
│   │   ├── train/           # Eğitim etiketleri (TXT - Yolo formatı)
│   │   └── val/             # Doğrulama etiketleri (TXT - Yolo formatı)
│   ├── label_map.pbtxt      # Sınıf haritası
│   └── data.yaml            # Yolo eğitim konfigürasyon dosyası
├── yolov5_model/            # YoloV5 dosyaları (yolo eğitimi için kullanacağımız model)
├── main.py                  # Flask uygulama kodu
├── xml_to_yolo_converter.py # XML'den Yolo'ya dönüştürücü script
└── README.md
```

### Adım Adım Dosya Düzenleme
1. **Veri Seti Düzenlemesi (traffic_signs_dataset_v1)**
   - `traffic_signs_dataset_v1` klasöründe:
     - **Görüntüleri düzenleme**: `train` ve `test` klasörlerinde bulunan `.jpg` dosyalarını `images/train` ve `images/val` klasörlerine taşıyın.
     - **XML etiket dosyalarını düzenleme**: `train` ve `test` klasörlerindeki `.xml` dosyalarını `labels/train` ve `labels/val` klasörlerine taşıyın. Bu dosyalar dönüşümden sonra Yolo formatında `.txt` dosyalarına dönüştürülecek.

2. **`data.yaml` Dosyasını Oluşturun**
   - `traffic_signs_dataset_v1` klasörüne `data.yaml` adlı bir dosya oluşturun. Bu dosya YoloV5 eğitiminde veri setinizin yolunu ve sınıflarını tanımlamak için kullanılacak. Örneğin:

   ```yaml
   train: ../traffic_signs_dataset_v1/images/train
   val: ../traffic_signs_dataset_v1/images/val

   nc: 19  # Sınıf sayısı
   names: ['Dur', 'Yol Ver', 'Sağa Git', ...]  # Sınıf isimleri (label_map.pbtxt içindeki sınıflar)
   ```

3. **XML Dönüşüm Script'i (xml_to_yolo_converter.py)**
   - Ana dizine (`TrafikIsaretlerTanima/`) `xml_to_yolo_converter.py` dosyasını ekleyin.
   - Bu script, `labels/train` ve `labels/val` klasörlerindeki `.xml` dosyalarını Yolo formatındaki `.txt` dosyalarına dönüştürecek ve `labels/train` ve `labels/val` klasörlerinde saklayacak.

   Örneğin, aşağıdaki script'i `xml_to_yolo_converter.py` olarak kaydedin ve çalıştırın:

   ```python
   import xml.etree.ElementTree as ET
   import os

   def convert_to_yolo_format(xml_file, output_txt_path, label_map, img_width, img_height):
       tree = ET.parse(xml_file)
       root = tree.getroot()

       with open(output_txt_path, "w") as out_file:
           for obj in root.iter('object'):
               class_name = obj.find('name').text
               if class_name not in label_map.values():
                   continue

               # Sınıf adını sınıf ID'sine dönüştürelim
               class_id = list(label_map.keys())[list(label_map.values()).index(class_name)]

               # Koordinatları alalım
               bndbox = obj.find('bndbox')
               xmin = int(bndbox.find('xmin').text)
               ymin = int(bndbox.find('ymin').text)
               xmax = int(bndbox.find('xmax').text)
               ymax = int(bndbox.find('ymax').text)

               # Yolo formatına uygun normalleştirilmiş koordinatları hesaplayalım
               x_center = ((xmin + xmax) / 2) / img_width
               y_center = ((ymin + ymax) / 2) / img_height
               width = (xmax - xmin) / img_width
               height = (ymax - ymin) / img_height

               # Sonucu dosyaya yazalım
               out_file.write(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")

   # Dizinleri belirtelim
   xml_dir = 'traffic_signs_dataset_v1/labels/train/'  # XML dosyalarının olduğu dizin
   output_txt_dir = 'traffic_signs_dataset_v1/labels/train/'  # Yolo formatındaki etiketlerin saklanacağı dizin

   os.makedirs(output_txt_dir, exist_ok=True)

   # Label map'i yükleyelim
   label_map = {
       0: 'Dur',
       1: 'Yol Ver',
       # Diğer sınıflar...
   }

   # Tüm XML dosyalarını işleyelim
   for xml_file in os.listdir(xml_dir):
       if xml_file.endswith('.xml'):
           xml_path = os.path.join(xml_dir, xml_file)
           image_file = xml_file.replace('.xml', '.jpg')
           img_path = os.path.join('traffic_signs_dataset_v1/images/train', image_file)

           # Görüntü boyutlarını XML dosyasından alalım
           tree = ET.parse(xml_path)
           root = tree.getroot()
           img_width = int(root.find('size/width').text)
           img_height = int(root.find('size/height').text)

           # Yolo formatına uygun TXT dosyasını oluşturalım
           output_txt_path = os.path.join(output_txt_dir, xml_file.replace('.xml', '.txt'))
           convert_to_yolo_format(xml_path, output_txt_path, label_map, img_width, img_height)
   ```

   Bu script'i çalıştırarak XML dosyalarınızı Yolo formatına dönüştürebilirsiniz.

4. **Modeli Eğitmek İçin YoloV5 Kullanımı**
   - **Eğitim Komutu**: `yolov5_model` dizinine giderek aşağıdaki komutu çalıştırabilirsiniz:

   ```bash
   python train.py --img 640 --batch 16 --epochs 50 --data ../traffic_signs_dataset_v1/data.yaml --weights yolov5s.pt --cache
   ```

   Bu komut, `data.yaml` dosyasına uygun olarak modeli eğitecek ve sonuçları `runs/train/exp` klasöründe saklayacaktır.

### Dosyaları Hangi Dizinde Çalıştırmalı?
- **`xml_to_yolo_converter.py`**: Ana dizinden (`TrafikIsaretlerTanima/`) çalıştırın. Bu script XML dosyalarınızı `labels/train` ve `labels/val` klasörlerine dönüştürecektir.
- **Yolo Eğitimi (`train.py`)**: `yolov5_model` klasörüne geçip eğitimi başlatın.

### Özet
1. **Veri Seti Düzenlemesi**: Görüntü ve etiketlerinizi ilgili klasörlere taşıyın.
2. **XML -> Yolo Formatına Dönüşüm**: `xml_to_yolo_converter.py` script'ini çalıştırarak XML dosyalarınızı Yolo formatına çevirin.
3. **YoloV5 Eğitimi**: `train.py` dosyasını kullanarak modeli eğitin.

Bu düzenlemeler ve adımlar, veri setinizi Yolo formatına uygun hale getirecek ve modeli başarılı bir şekilde eğitmenizi sağlayacaktır. Yardımcı olmamı istediğiniz başka bir konu olursa lütfen bana bildirin.
