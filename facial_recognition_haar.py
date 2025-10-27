import cv2 
import os
import numpy as np
import winsound
from flask import Flask, jsonify
from threading import Thread

# Flask app setup
app = Flask(__name__)
recognized_faces = []  # Store recognized faces

@app.route('/get_logs', methods=['GET'])
def get_logs():
    return jsonify({"logs": recognized_faces})

def run_flask():
    app.run(host='127.0.0.1', port=8000)

# Start Flask in a separate thread
flask_thread = Thread(target=run_flask)
flask_thread.daemon = True
flask_thread.start()

# Path to the dataset
known_faces_path = "known_faces"

# Initialize LBPHFaceRecognizer_create()
recognizer = cv2.face.LBPHFaceRecognizer_create()

# Mapping IDs to names
id_to_name = {
    1: "Richie"
}

# Lists to hold training data and labels
training_data = []
labels = []

# Function to read images and labels 
def prepare_training_data(known_faces_path):
    print("Preparing training data...")
    for label_folder in os.listdir(known_faces_path):
        label_path = os.path.join(known_faces_path, label_folder)
        if not os.path.isdir(label_path):
            continue

        label = int(label_folder)  # Convert folder name to integer label
        print(f"Processing label: {label}")  # Debugging print
        for image_file in os.listdir(label_path):
            image_path = os.path.join(label_path, image_file)
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)  # Load in grayscale
            if image is None:
                print(f"Skipping {image_file}, could not load.")
                continue

            image = cv2.resize(image, (200, 200))
            training_data.append(image)
            labels.append(label)
            print(f"Loaded {image_file} for label {label}")

# Prepare the training data
prepare_training_data(known_faces_path)

# Check if we have data
if len(training_data) == 0 or len(labels) == 0:
    print("No faces detected in the training dataset.")
    exit()

print(f"Training data size: {len(training_data)}, Labels: {len(labels)}")

# Convert to numpy arrays
training_data = np.array(training_data)
labels = np.array(labels)

# Train the recognizer 
recognizer.train(training_data, labels)

# Save the trained model
recognizer.save("trained_model.yml")
print("Model trained and saved!")

# Load the trained model
recognizer.read("trained_model.yml")
print("Model loaded again for verification")

# Load the Haar Cascade classifier 
face_cascade = cv2.CascadeClassifier('C:/Facial_Recognition_Alert_System/haarcascade_frontalface_default.xml')

# Start video capture
cap = cv2.VideoCapture(1)

while True:
    ret, frame = cap.read()
    if not ret:
        break

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

        # Limit the size of the logs to avoid memory issues
        if len(recognized_faces) > 100:
            recognized_faces.pop(0)

        text = f"{name}, Confidence: {confidence:.2f}" if confidence < 100 else "Unknown face"
        cv2.putText(frame, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

    cv2.imshow('Face Detection', frame)
    if cv2.waitKey(500) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()