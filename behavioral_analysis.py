import cv2
import os
import numpy as np
import winsound
from flask import Flask, jsonify
from threading import Thread, Lock
import time
from ultralytics import YOLO

# Flask app setup
app = Flask(__name__)
recognized_faces = []  # Store recognized faces
behavior_logs = []  # Store behavior logs

@app.route('/get_logs', methods=['GET'])
def get_logs():
    return jsonify({"logs": recognized_faces})

@app.route('/get_behavior_logs', methods=['GET'])
def get_behavior_logs():
    return jsonify({"behavior_logs": behavior_logs})

def run_flask():
    app.run(host='127.0.0.1', port=8000)

# Global shared frame and lock
latest_frame = None
frame_lock = Lock()

# Camera reader thread
def camera_reader():
    global latest_frame
    cap = cv2.VideoCapture(1)

    while True:
        ret, frame = cap.read()
        if not ret:
            continue
        with frame_lock:
            latest_frame = frame.copy()
        time.sleep(0.01)

    cap.release()

# Facial recognition system
def face_recognition_system():
    global latest_frame
    known_faces_path = "known_faces"
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    id_to_name = {1: "Richie"}
    training_data = []
    labels = []

    def prepare_training_data(known_faces_path):
        for label_folder in os.listdir(known_faces_path):
            label_path = os.path.join(known_faces_path, label_folder)
            if not os.path.isdir(label_path):
                continue

            label = int(label_folder)
            for image_file in os.listdir(label_path):
                image_path = os.path.join(label_path, image_file)
                image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
                if image is None:
                    continue

                image = cv2.resize(image, (200, 200))
                training_data.append(image)
                labels.append(label)

    prepare_training_data(known_faces_path)

    if len(training_data) == 0 or len(labels) == 0:
        print("No faces detected in the training dataset.")
        return

    training_data = np.array(training_data)
    labels = np.array(labels)
    recognizer.train(training_data, labels)
    recognizer.save("trained_model.yml")
    recognizer.read("trained_model.yml")

    face_cascade = cv2.CascadeClassifier('C:/Facial_Recognition_Alert_System/haarcascade_frontalface_default.xml')

    while True:
        with frame_lock:
            if latest_frame is None:
                continue
            frame = latest_frame.copy()

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=3, minSize=(50, 50))

        for (x, y, w, h) in faces:
            face_roi = gray[y:y+h, x:x+w]
            face_roi = cv2.resize(face_roi, (200, 200))
            face_roi = cv2.equalizeHist(face_roi)

            label, confidence = recognizer.predict(face_roi)

            if confidence < 100:
                name = id_to_name.get(label, "Unknown")
                recognized_faces.append({"name": name, "confidence": confidence})
            else:
                recognized_faces.append({"name": "Unknown", "confidence": confidence})

            if len(recognized_faces) > 100:
                recognized_faces.pop(0)

            text = f"{name}, Confidence: {confidence:.2f}" if confidence < 100 else "Unknown face"
            cv2.putText(frame, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

        cv2.imshow('Face Detection', frame)
        if cv2.waitKey(500) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()

# Behavior analysis system
def behavior_analysis_system():
    global latest_frame
    model = YOLO("yolov8m-pose.pt")
    prev_coords = None
    last_logged_posture = {}  # person_index -> timestamp

    while True:
        with frame_lock:
            if latest_frame is None:
                continue
            frame = latest_frame.copy()

        results = model.predict(frame, conf=0.5)
        annotated_frame = results[0].plot()

        keypoints = results[0].keypoints
        if keypoints is not None and keypoints.xy is not None:
            kp_coords = keypoints.xy.cpu().numpy()

            if hasattr(keypoints, 'conf') and keypoints.conf is not None:
                kp_scores = keypoints.conf.cpu().numpy()

                for i in range(len(kp_coords)):
                    label = ""

                    # Posture detection
                    if kp_scores[i, 6] > 0.5 and kp_scores[i, 12] > 0.5:
                        y_shoulder = kp_coords[i, 6, 1]
                        y_hip = kp_coords[i, 12, 1]
                        y_diff = abs(y_shoulder - y_hip)

                        if y_diff < 80:
                            label = "Sitting"
                            now = time.time()
                            if i not in last_logged_posture or (now - last_logged_posture[i]) > 30:
                                behavior_logs.append({
                                    "event": f"Person {i+1} was sitting idle",
                                    "timestamp": time.strftime('%I:%M %p')
                                })
                                last_logged_posture[i] = now
                        else:
                            label = "Standing"

                        x = int(kp_coords[i, 6, 0])
                        y = int(kp_coords[i, 6, 1]) - 10
                        cv2.putText(annotated_frame, label, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

                    # Throw-like movement detection
                    if prev_coords is not None:
                        try:
                            prev_x, prev_y = prev_coords[i, 10]
                            curr_x, curr_y = kp_coords[i, 10]
                            dx = abs(curr_x - prev_x)
                            dy = abs(curr_y - prev_y)

                            if dx > 40 or dy > 40:
                                behavior_logs.append({
                                    "event": f"Worker #{i+1} exhibited throw-like motion",
                                    "timestamp": time.strftime('%I:%M %p')
                                })
                                cv2.putText(annotated_frame, "Throw-like movement detected!", (50, 50),
                                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)
                        except IndexError:
                            pass

            prev_coords = kp_coords

        if len(behavior_logs) > 100:
            behavior_logs.pop(0)

        cv2.imshow("Behavior Detection", annotated_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()

if __name__ == "__main__":
    Thread(target=run_flask, daemon=True).start()
    Thread(target=camera_reader, daemon=True).start()
    Thread(target=face_recognition_system, daemon=True).start()
    Thread(target=behavior_analysis_system, daemon=True).start()

    while True:
        try:
            time.sleep(1)
        except KeyboardInterrupt:
            print("Exiting system...")
            break
