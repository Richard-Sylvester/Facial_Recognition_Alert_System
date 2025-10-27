#import cv2
import face_recognition
import os
import numpy as np
from sklearn.neighbors import KDTree

# Load known faces
known_face_encodings = []
known_face_names = []

print("Loading known faces...")
known_faces_path = "known_faces"
for file_name in os.listdir(known_faces_path):
    # Load image and encode it 
    image_path = os.path.join(known_faces_path, file_name)
    image = face_recognition.load_image_file(image_path)
    encoding = face_recognition.face_encodings(image)

    if encoding:
    # Append to known face list
        known_face_encodings.append(encoding[0])  # Use the first encoding 
        known_face_names.append(file_name.split('.')[0]) # Use file name as name 
    else:
        print(f"Warning: No faces found in {file_name}. Please use clear images with a single face.")

# Build KD-Tree for fast nearest neighbor search
kd_tree = KDTree(known_face_encodings)

print(f"loaded {len(known_face_names)} known faces.")

# Initialize video capture
video_capture = cv2.VideoCapture(0)

# Set the webcam resolution to a lower value
video_capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)  # Set width
video_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)  # Set height

print("Starting video stream...")
frame_count = 0

while True:
    # Grab a frame from the webcam
    ret, frame = video_capture.read()
    if not ret:
        print("Failed to grab frame. Exiting...")
        break

    # Resize frame for faster processing 
    small_frame = cv2.resize(frame, (0,0), fx=0.15, fy=0.15)
    rgb_frame = frame[:, :, ::-1]  # Converst BGR to RGB

    # Process every 30th frame (adjust if necessary)
    frame_count += 1
    if frame_count % 30 != 0:
        cv2.imshow('Facial recognition', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        continue

# def recognize_faces(rgb_frame, known_face_encodings, known_face_names):
    face_locations = face_recognition.face_locations(rgb_frame, model="hog")[:1]  # Process only first face
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
    # results = []

    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        distances, indices = kd_tree.query([face_encoding], k=1)
        name = "Unknown"

        # if the closest match is below a distance threshold, consider it a match
        if distances[0][0] < 0.6:
            name = known_face_names[indices[0][0]]

        # Scale the face locations    
        top *= 2
        right *= 2
        bottom *= 2
        left *= 2

        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
        cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 255, 0), 1)
        

    # Display the frame
    cv2.imshow('Facial Recognition', frame)

    # Break loop on 'q' key
    if cv2.waitKey(1) & 0XFF == ord('q'):
        break

# Release video capture and close windows
video_capture.release()
cv2.destroyAllWindows()