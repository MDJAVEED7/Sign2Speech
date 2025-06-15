import cv2
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier
import numpy as np
import math
from flask import Flask, render_template, Response, redirect, url_for
from flask_socketio import SocketIO

app = Flask(__name__)
socketio = SocketIO(app)

cap = None
video_running = False

def release_capture():
    global cap
    if cap:
        cap.release()
        cap = None

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/main')
def main():
    return render_template('main.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/contact')
def contact():
    return render_template('contact.html')

@app.route('/video')
def video():
    global video_running
    video_running = True
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/close')
def close():
    global video_running
    video_running = False
    release_capture()
    return redirect(url_for('index'))

@socketio.on('update')
def handle_update(data):
    detected_label = data.get('detectedLabel', '')
    print(f'Detected Label: {detected_label}')
    socketio.emit('update', {'detectedLabel': detected_label})

def generate_frames():
    global cap, video_running

    if cap is None or not cap.isOpened():
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("❌ Error: Could not open camera.")
            return

    detector = HandDetector(maxHands=1)
    classifier = Classifier("Model/keras_model.h5", "Model/labels.txt")

    offset = 20
    imgSize = 300
    labels = ["see", "HELLO", "Any Question", "superb", "rocked", "like", "dislike", "help", "one", "Right"]

    while video_running and cap.isOpened():
        success, img = cap.read()
        if not success:
            print("❌ Error: Failed to capture frame from camera.")
            break

        imgOutput = img.copy()
        hands, img = detector.findHands(img)

        if hands:
            hand = hands[0]
            x, y, w, h = hand['bbox']
            imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255
            imgCrop = img[y - offset:y + h + offset, x - offset:x + w + offset]

            if imgCrop.size > 0:
                aspectRatio = h / w

                try:
                    if aspectRatio > 1:
                        k = imgSize / h
                        wCal = math.ceil(k * w)
                        imgResize = cv2.resize(imgCrop, (wCal, imgSize))
                        wGap = math.ceil((imgSize - wCal) / 2)
                        imgWhite[:, wGap:wGap + wCal] = imgResize
                    else:
                        k = imgSize / w
                        hCal = math.ceil(k * h)
                        imgResize = cv2.resize(imgCrop, (imgSize, hCal))
                        hGap = math.ceil((imgSize - hCal) / 2)
                        imgWhite[hGap:hGap + hCal, :] = imgResize

                    prediction, index = classifier.getPrediction(imgWhite, draw=False)

                    cv2.rectangle(imgOutput, (x - offset, y - offset - 50),
                                  (x - offset + 90, y - offset), (255, 0, 255), cv2.FILLED)
                    cv2.putText(imgOutput, labels[index], (x, y - 26),
                                cv2.FONT_HERSHEY_COMPLEX, 1.7, (255, 255, 255), 2)
                    cv2.rectangle(imgOutput, (x - offset, y - offset),
                                  (x + w + offset, y + h + offset), (255, 0, 255), 4)
                except Exception as e:
                    print(f"⚠️ Error during processing: {e}")

        ret, buffer = cv2.imencode('.jpg', imgOutput)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    release_capture()

if __name__ == "__main__":
    socketio.run(app, debug=True, use_reloader=False, port=5000, allow_unsafe_werkzeug=True)
