from flask import Flask, render_template, Response, redirect, url_for, request, flash
import cv2
import face_recognition
import numpy as np
import csv
from datetime import datetime
import pickle
import secrets

app = Flask(__name__)
app.secret_key = secrets.token_hex(16)

# Global variable to control camera feed
camera_active = False
video_capture = None

# Load encodings from file
with open("encodings.pkl", "rb") as f:
    known_face_encodings, known_face_names, known_emp_id = pickle.load(f)

background_image = cv2.imread("photos/camback.png")
background_width, background_height = 600, 500
background_resized = cv2.resize(background_image, (background_width, background_height))

camera_feed_width, camera_feed_height = 525, 305
x_offset, y_offset = 31, 67

students = list(set(known_face_names))
displayed_names = {}

def get_csv_file_path():
    """Generate the CSV file path based on the current date."""
    current_date = datetime.now().strftime("%Y-%m-%d")
    return f"attendance_{current_date}.csv"

def find_working_camera_index():
    """Attempts to find an available camera index."""
    for index in range(5):  # Try up to 5 camera indices
        cap = cv2.VideoCapture(index)
        if cap.isOpened():
            cap.release()
            return index
    return -1  # No available camera

def generate_frames():
    global camera_active, video_capture
    try:
        while camera_active:
            if video_capture is None or not video_capture.isOpened():
                print("Camera not opened.")
                break

            ret, frame = video_capture.read()
            if not ret:
                print("Failed to capture frame")
                camera_active = False
                break

            small_frame = cv2.resize(frame, (camera_feed_width, camera_feed_height))
            overlay_frame = background_resized.copy()
            overlay_frame[y_offset:y_offset + camera_feed_height, x_offset:x_offset + camera_feed_width] = small_frame

            rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
            face_locations = face_recognition.face_locations(rgb_small_frame)
            face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
            
            detected_names = set()
            detected_emp_id = set()

            for face_encoding in face_encodings:
                face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
                best_match_index = np.argmin(face_distances)
                
                if face_distances[best_match_index] < 0.6:
                    name = known_face_names[best_match_index]
                    detected_names.add(name)
                    emp_id = known_emp_id[best_match_index]
                    detected_emp_id.add(emp_id)

                    if name in students:
                        students.remove(name)
                        current_time = datetime.now().strftime("%H:%M:%S")
                        csv_file_path = get_csv_file_path()
                        with open(csv_file_path, "a+", newline="") as csv_file:
                            lnwriter = csv.writer(csv_file)
                            lnwriter.writerow([name, emp_id, current_time])
                        print(f"Recorded {name} at {current_time}")

            if not detected_names:
                font = cv2.FONT_HERSHEY_SIMPLEX
                bottomLeftCornerOfText = (142, 450)
                fontScale = 1
                fontColor = (0, 0, 255)
                thickness = 2
                lineType = 2
                cv2.putText(overlay_frame, "No Match Found", bottomLeftCornerOfText, font, fontScale, fontColor, thickness, lineType)
            
            for name in detected_names:
                displayed_names[name] = True

            for name in displayed_names.keys():
                if name in detected_names:
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    bottomLeftCornerOfText = (170, 450 + 30 * list(displayed_names.keys()).index(name))
                    fontScale = 1
                    fontColor = (0, 255, 0)
                    thickness = 2
                    lineType = 2
                    cv2.putText(overlay_frame, name + " Present", bottomLeftCornerOfText, font, fontScale, fontColor, thickness, lineType)

            ret, buffer = cv2.imencode('.jpg', overlay_frame)
            if not ret:
                print("Failed to encode frame")
                camera_active = False
                break
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
    except Exception as e:
        print(f"Error in generating frames: {e}")
        camera_active = False
    finally:
        if video_capture is not None:
            video_capture.release()
            video_capture = None

@app.route('/view_attendance')
def view_attendance():
    csv_file_path = get_csv_file_path()
    attendance_records = []
    try:
        with open(csv_file_path, 'r') as csv_file:
            csv_reader = csv.reader(csv_file)
            for row in csv_reader:
                attendance_records.append(row)
    except FileNotFoundError:
        attendance_records = []

    return render_template('view_attendance.html', records=attendance_records)

@app.route('/take_attendance')
def take_attendance():
    return render_template('take_attendance.html')

@app.route('/video_feed')
def video_feed():
    global camera_active, video_capture
    camera_active = True
    camera_index = find_working_camera_index()
    if camera_index == -1:
        flash("No available camera found.")
        return redirect(url_for('home'))
    
    video_capture = cv2.VideoCapture(camera_index)
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/stop_video_feed', methods=['GET'])
def stop_video_feed():
    global camera_active
    camera_active = False
    return redirect(url_for('home'))

@app.route('/')
def home():
    return render_template('home.html')

if __name__ == '__main__':
    app.run(debug=True)
