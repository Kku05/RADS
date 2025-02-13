import face_recognition
import pickle
import os
import csv

def process_images(image_paths, label, emp_id):
    encodings = []
    names = []
    emp_ids = []
    for image_path in image_paths:
        try:
            image = face_recognition.load_image_file(image_path)
            image_encodings = face_recognition.face_encodings(image)
            for encoding in image_encodings:
                encodings.append(encoding)
                names.append(label)
                emp_ids.append(emp_id)
        except Exception as e: 
            print(f"Error loading {image_path}: {e}")
    return encodings, names, emp_ids

def load_known_faces_from_csv(csv_file):
    known_face_encodings = []
    known_face_names = []
    known_emp_ids = []

    with open(csv_file, mode='r') as file:
        reader = csv.DictReader(file)
        for row in reader:
            label = row['label']
            emp_id = row['emp_id']
            image_path = row['image_path']
            encodings, names, emp_ids = process_images([image_path], label, emp_id)
            known_face_encodings.extend(encodings)
            known_face_names.extend(names)
            known_emp_ids.extend(emp_ids)

    return known_face_encodings, known_face_names, known_emp_ids

if __name__ == "__main__":
    csv_file = "photos.csv"  
    known_face_encodings, known_face_names, known_emp_ids = load_known_faces_from_csv(csv_file)
    
    with open("encodings.pkl", "wb") as f:
        pickle.dump((known_face_encodings, known_face_names, known_emp_ids), f)