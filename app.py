import cv2
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier
import numpy as np
import math
from flask import Flask, render_template, Response, redirect, url_for
from flask_socketio import SocketIO

app = Flask(__name__)
socketio = SocketIO(app)

# Declare the video capture variable outside the generator function
cap = None
video_running = False  # Flag to control video streaming

# Function to release the video capture
def release_capture():
    global cap
    if cap:
        cap.release()

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
    video_running = False  # Set the flag to stop video streaming
    release_capture()  # Release the video capture when the loop exits
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
            print("Error: Could not open camera.")
            release_capture()
            return

    detector = HandDetector(maxHands=1)
    classifier = Classifier("Model/keras_model.h5", "Model/labels.txt")

    offset = 20
    imgSize = 300

    labels = ["see", "HELLO", "Any Question", "superb", "rocked", "like", "dislike", "help", "one", "Right"]

    while cap.isOpened() and video_running:
        success, img = cap.read()
        if not success:
            print("Error: Could not read frame.")
            break

        imgOutput = img.copy()
        hands, img = detector.findHands(img)

        if hands:
            hand = hands[0]
            x, y, w, h = hand['bbox']

            imgCrop = img[y - offset:y + h + offset, x - offset:x + w + offset]

            if not imgCrop.size == 0 and w > 0 and h > 0:
                aspectRatio = h / w

                if aspectRatio > 1:
                    k = imgSize / h
                    wCal = math.ceil(k * w)
                    imgResize = cv2.resize(imgCrop, (wCal, imgSize))
                    imgResizeShape = imgResize.shape
                    wGap = math.ceil((imgSize - wCal) / 2)
                    imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255
                    imgWhite[:, wGap:wGap + wCal] = imgResize
                    prediction, index = classifier.getPrediction(imgWhite, draw=False)

                else:
                    k = imgSize / w
                    hCal = math.ceil(k * h)
                    imgResize = cv2.resize(imgCrop, (imgSize, hCal))
                    imgResizeShape = imgResize.shape
                    hGap = math.ceil((imgSize - hCal) / 2)
                    imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255
                    imgWhite[hGap:hGap + hCal, :] = imgResize
                    prediction, index = classifier.getPrediction(imgWhite, draw=False)

                cv2.rectangle(imgOutput, (x - offset, y - offset - 50),
                              (x - offset + 90, y - offset - 50 + 50), (255, 0, 255), cv2.FILLED)
                cv2.putText(imgOutput, labels[index], (x, y - 26), cv2.FONT_HERSHEY_COMPLEX, 1.7, (255, 255, 255), 2)
                cv2.rectangle(imgOutput, (x - offset, y - offset),
                              (x + w + offset, y + h + offset), (255, 0, 255), 4)

        ret, buffer = cv2.imencode('.jpg', imgOutput)
        img_output = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + img_output + b'\r\n')

    release_capture()  # Release the video capture when the loop exits

if __name__ == "__main__":
    socketio.run(app, debug=True, use_reloader=False, port=5000, allow_unsafe_werkzeug=True)
