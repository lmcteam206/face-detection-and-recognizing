import cv2
import os
import numpy as np
import art
from art import *
from datetime import datetime
import sqlite3

Rody = 0
Basel = 0
file = open("School_attendance _Database.txt","w")
now = datetime.now()

# Function to load images and labels from a directory
def load_images_from_folder(folder):
    images = []
    labels = []
    label_dict = {}  # Dictionary to map integer labels to names
    label_id = 0  # Initialize label ID counter
    for filename in os.listdir(folder):
        path = os.path.join(folder, filename)
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is not None:
            images.append(img)
            # Extract name from filename
            name = os.path.splitext(filename)[0]
            # Check if name already exists in label dictionary
            if name not in label_dict:
                label_dict[name] = label_id
                label_id += 1
            labels.append(label_dict[name])  # Use integer label
    return images, labels, label_dict

# Load images and labels
images, labels, label_dict = load_images_from_folder('dataset')

# Convert labels to numpy array
labels = np.array(labels)

# Create LBPH recognizer
recognizer = cv2.face.LBPHFaceRecognizer_create()

# Train the recognizer
recognizer.train(images, labels)

# Initialize webcam
video_capture = cv2.VideoCapture(0)

while True:
    # Capture frame-by-frame
    ret, frame = video_capture.read()

    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the frame
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # For each detected face, perform recognition
    for (x, y, w, h) in faces:
        # Extract the face ROI
        face_roi = gray[y:y+h, x:x+w]

        # Perform recognition
        label_id, confidence = recognizer.predict(face_roi)

        # If the confidence is low, classify as unknown
        if confidence < 88:
            print("#############################################################################################################")
            name = "Unknown"
            tprint(name)
           
        else:
            print("#############################################################################################################")
            name = list(label_dict.keys())[list(label_dict.values()).index(label_id)]  
            name = list(label_dict.keys())[list(label_dict.values()).index(label_id)]  
            name = list(label_dict.keys())[list(label_dict.values()).index(label_id)]  
            name = list(label_dict.keys())[list(label_dict.values()).index(label_id)]  
            name = list(label_dict.keys())[list(label_dict.values()).index(label_id)]  
            name = list(label_dict.keys())[list(label_dict.values()).index(label_id)]  
            name = list(label_dict.keys())[list(label_dict.values()).index(label_id)]  
            name = list(label_dict.keys())[list(label_dict.values()).index(label_id)]  # Get name corresponding to label
            print(now.strftime('%Y/%m/%d %I:%M:%S'))
            print(name)

            if (name == "Basel"):
                if (Basel == 0):
                    file.write("\n Student :::::::::::::::::::::::::::::::::::::::: \n")
                    file.write(now.strftime('%Y/%m/%d %I:%M:%S\n'))
                    file.write(name)
                    # end the list
                    Basel = 1

            if (name == "Rody"):
                if (Rody == 0):
                    file.write("\n Student :::::::::::::::::::::::::::::::::::::::: \n")
                    file.write(now.strftime('%Y/%m/%d %I:%M:%S\n'))
                    file.write(name)
                    # end the list  
                    Rody = 1                 

        # Draw a rectangle around the face
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

        # Draw the name next to the face
        cv2.putText(frame, name, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # Display the resulting frame
    cv2.imshow('School_attendance_Database', frame)

    # Break the loop when 'q' is pressed
    if cv2.waitKey(2) & 0xFF == ord('q'):
        break

# Release the webcam
video_capture.release()
cv2.destroyAllWindows()
