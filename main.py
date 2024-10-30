import torch
from flask import Flask, render_template, Response
import cv2
import pyttsx3
import time
from multiprocessing import Process, Queue

# Flask application
app = Flask(__name__)

# Initialize the camera with OpenCV
camera = cv2.VideoCapture(1)
camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

# Load the YoloV5 model
model = torch.hub.load('yolov5_model', 'custom', path='yolov5_model/weights/best.pt', source='local')

# Variables to manage timing
last_console_print_time = 0

# Queue to communicate detected labels to the speech process
label_queue = Queue()

def speech_process(label_queue):
    engine = pyttsx3.init()
    engine.setProperty('voice', 'tr')  # Turkish language support

    last_speech_time = time.time()
    detected_labels = set()
    while True:
        # Collect labels for 3 seconds
        try:
            while True:
                # Non-blocking get with timeout
                label = label_queue.get(timeout=0.1)
                detected_labels.add(label)
        except:
            # Timeout occurred, no more labels in the queue
            pass

        current_time = time.time()
        if current_time - last_speech_time >= 3:
            if detected_labels:
                # Read the first detected label
                label_to_speak = next(iter(detected_labels))
                engine.say(label_to_speak)
                engine.runAndWait()
                detected_labels.clear()
            last_speech_time = current_time
        time.sleep(0.1)  # Slight delay to prevent CPU overuse

def generate_frames():
    global last_console_print_time
    while True:
        # Capture frame from the camera
        success, frame = camera.read()
        if not success:
            print("Unable to receive frame from the camera.")
            break
        else:
            # Process the frame with YoloV5
            results = model(frame)

            # Extract labels and coordinates
            labels, cords = results.xyxyn[0][:, -1], results.xyxyn[0][:, :-1]
            current_time = time.time()
            y_offset = 50  # Initial y-coordinate for labels on the frame

            for label, cord in zip(labels, cords):
                x1, y1, x2, y2, conf = cord
                if conf > 0.3:  # Filter based on confidence score
                    x1, y1, x2, y2 = int(x1 * frame.shape[1]), int(y1 * frame.shape[0]), int(x2 * frame.shape[1]), int(y2 * frame.shape[0])
                    label_name = results.names[int(label)]

                    # Draw bounding box and label on the frame
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, label_name, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)

                    # Add label to the queue for speech
                    label_queue.put(label_name)

                    # Print to console every 1 second
                    if current_time - last_console_print_time >= 1:
                        print(f"Detected object: {label_name}")
                        last_console_print_time = current_time

                    # Display labels on the web page without overlapping
                    cv2.putText(frame, f"Detection: {label_name}", (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    y_offset += 30  # Move to the next line for the next label

            # Send the frame to the browser
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
    # Start the speech process
    p = Process(target=speech_process, args=(label_queue,))
    p.start()

    try:
        # Start the Flask server
        print("Starting Flask server... Go to http://127.0.0.1:5000 in your browser.")
        app.run(host='0.0.0.0', port=5000, debug=False)
    finally:
        # Terminate the speech process when the app is closed
        p.terminate()
        p.join()
