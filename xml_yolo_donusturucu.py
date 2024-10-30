import xml.etree.ElementTree as ET
import os


def convert_to_yolo_format(xml_file, output_txt_path, label_map, img_width, img_height):
    try:
        tree = ET.parse(xml_file)
        root = tree.getroot()

        with open(output_txt_path, "w") as out_file:
            for obj in root.iter('object'):
                class_name = obj.find('name').text.strip()  # Sınıf adını alırken baştaki ve sondaki boşlukları temizle
                if class_name not in label_map.values():
                    print(f"Uyarı: Sınıf '{class_name}' etiket haritasında bulunamadı. Geçiliyor...")
                    continue  # Eğer sınıf adını bulamazsak atla

                # Sınıf adını sınıf ID'sine dönüştürelim
                class_id = list(label_map.keys())[list(label_map.values()).index(class_name)]

                # Koordinatları alalım
                bndbox = obj.find('bndbox')
                if bndbox is None:
                    print(f"Uyarı: Bndbox etiketi XML dosyasında bulunamadı. Geçiliyor: {xml_file}")
                    continue

                try:
                    xmin = int(float(bndbox.find('xmin').text))
                    ymin = int(float(bndbox.find('ymin').text))
                    xmax = int(float(bndbox.find('xmax').text))
                    ymax = int(float(bndbox.find('ymax').text))
                except ValueError as e:
                    print(f"Hata: Koordinat değerleri okunurken hata oluştu. Geçiliyor: {xml_file}. Hata: {e}")
                    continue

                # Yolo formatına uygun normalleştirilmiş koordinatları hesaplayalım
                x_center = ((xmin + xmax) / 2) / img_width
                y_center = ((ymin + ymax) / 2) / img_height
                width = (xmax - xmin) / img_width
                height = (ymax - ymin) / img_height

                # Eğer koordinatlar geçerli değilse atlayın
                if not (0 <= x_center <= 1 and 0 <= y_center <= 1 and 0 <= width <= 1 and 0 <= height <= 1):
                    print(f"Uyarı: Koordinatlar normalleştirilmiş aralıkta değil. Geçiliyor: {xml_file}")
                    continue

                # Sonucu dosyaya yazalım
                out_file.write(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")

    except ET.ParseError:
        print(f"Hata: {xml_file} dosyası XML olarak parse edilemedi.")
    except Exception as e:
        print(f"Hata: {xml_file} dosyası işlenirken hata oluştu: {e}")


# Dizinleri belirtelim
xml_dir = 'traffic_signs_dataset_v1/labels/train/'  # XML dosyalarının olduğu dizin
output_txt_dir = 'traffic_signs_dataset_v1/labels/train_yolo/'  # Yolo formatındaki etiketlerin saklanacağı dizin

os.makedirs(output_txt_dir, exist_ok=True)

# Label map'i yükleyelim
label_map = {
    0: "30km_h Hiz Limiti (Speed limit)",
    1: "50km_h Hiz Limiti (Speed limit)",
    2: "60km_h Hiz Limiti (Speed limit)",
    3: "70km_h Hiz Limiti (Speed limit)",
    4: "90km_h Hiz Limiti (Speed limit)",
    5: "Bir Sonraki Kavsakta Gecis Hakki (Right-of-way at the next intersection)",
    6: "Buzlanma (Beware of ice/snow)",
    7: "Dur (Stop)",
    8: "Duz veya Saga Gidis (Go straight or right)",
    9: "Duz veya Sola Gidis (Go straight or left)",
    10: "Engebeli Yol (Bumpy road)",
    11: "Gecis Yasagi Sonu (End of no passing)",
    12: "Gecis Yasagi (No passing)",
    13: "Genel Uyari (General caution)",
    14: "Saga Surekli Tehlikeli Viraj (Double curve-Right)",
    15: "Kaygan yol (Slippery road)",
    16: "Oncelikli Yol (Priority road)",
    17: "Saga Tehlikeli Viraj (Dangerous curve to the right)",
    18: "Sagdan Gidiniz (Keep right)",
    19: "Sola Tehlikeli Viraj (Dangerous curve to the left)",
    20: "Soldan Gidiniz (Keep left)",
    21: "Trafik Isiklari (Traffic signals)",
    22: "Yayalar (Pedestrians)",
    23: "Yol Calismasi (Road work)",
    24: "Yol Ver (Yield)",
    25: "Zorunlu Doner Kavsak (Roundabout mandatory)",
    26: "U Donusu (U-turn)",
    27: "U Donusu Yasak (U Turn is not allowed)",
    28: "Gevsek Yamac (Loose Slope)",
    29: "4_80 m den Yuksek Arac giremez (Vehicle higher than 4_80 m cannot enter)",
    30: "Gevsek Malzemeli Yol (Loose Material Road)",
    31: "Ehli Hayvan Gecebilir (Animal crossing)",
    32: "Yaya Gecidi (Crosswalk)",
    33: "Duraklamak ve Parketmek Yasaktir (No, stoping and parking)",
    34: "Her Iki Yandan gidiniz (Go from both sides)",
    35: "Sola Surekli Tehlikeli Viraj (Double curve-Right)",
    36: "Okul Gecidi (School passing)",
    37: "EDS",
    38: "Egimli Yol (Slop)",
    39: "Yol Daralmasi (Road narrowing)",
    40: "Sagdan Ana Yola Giris (Entry from Right to Main Road)"
}

# Tüm XML dosyalarını işleyelim
for xml_file in os.listdir(xml_dir):
    if xml_file.endswith('.xml'):
        xml_path = os.path.join(xml_dir, xml_file)
        image_file = xml_file.replace('.xml', '.jpg')
        img_path = os.path.join('traffic_signs_dataset_v1/images/train', image_file)

        # Görüntü boyutlarını XML dosyasından alalım
        try:
            tree = ET.parse(xml_path)
            root = tree.getroot()
            img_width = int(root.find('size/width').text)
            img_height = int(root.find('size/height').text)

            # Yolo formatına uygun TXT dosyasını oluşturalım
            output_txt_path = os.path.join(output_txt_dir, xml_file.replace('.xml', '.txt'))
            convert_to_yolo_format(xml_path, output_txt_path, label_map, img_width, img_height)

        except Exception as e:
            print(f"Hata: {xml_file} dosyasından görüntü boyutları okunurken hata oluştu: {e}")