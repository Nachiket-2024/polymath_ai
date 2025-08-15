import os
import cv2
import numpy as np
import requests
import threading
from keras_facenet import FaceNet
from sklearn.metrics.pairwise import cosine_similarity
from concurrent.futures import ThreadPoolExecutor

# Suppress TensorFlow logs
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Configuration
FACE_DB_PATH = r"database_images"
THRESHOLD = 0.7

# Global variables with thread locks
face_db = {}
current_face = "Unknown"
ip_info = {}
face_lock = threading.Lock()

# Initialize FaceNet
embedder = FaceNet()

# IP Information Function
def get_ip_details():
    try:
        ip = requests.get("https://api64.ipify.org?format=json").json()["ip"]
        response = requests.get(f"http://ip-api.com/json/{ip}")
        return response.json() if response.status_code == 200 else {}
    except Exception as e:
        return {"error": str(e)}

# Face Database Loading with Multithreading
def process_image(img_path):
    img = cv2.imread(img_path)
    if img is None:
        return None
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    faces = embedder.extract(img, threshold=0.90)
    if faces:
        return faces[0]['embedding']
    return None

def load_face_database():
    global face_db
    print("Loading face database...")
    with ThreadPoolExecutor() as executor:
        for person in os.listdir(FACE_DB_PATH):
            person_dir = os.path.join(FACE_DB_PATH, person)
            if os.path.isdir(person_dir):
                img_paths = [os.path.join(person_dir, f) for f in os.listdir(person_dir)]
                embeddings = list(filter(lambda x: x is not None, executor.map(process_image, img_paths)))
                if embeddings:
                    with face_lock:
                        face_db[person] = np.mean(embeddings, axis=0)
    print(f"Loaded {len(face_db)} face embeddings")

# Face Recognition Function
def recognize_faces(frame):
    global current_face
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    faces = embedder.extract(rgb_frame, threshold=0.90)
    
    best_match = ("Unknown", 0)
    for face in faces:
        embedding = face['embedding']
        for name, db_emb in face_db.items():
            similarity = cosine_similarity([embedding], [db_emb])[0][0]
            if similarity > best_match[1] and similarity > THRESHOLD:
                best_match = (name, similarity)
    
    with face_lock:
        current_face = best_match[0]
    
    # Draw face annotations
    for face in faces:
        x, y, w, h = face['box']
        color = (0, 255, 0) if best_match[0] != "Unknown" else (0, 0, 255)
        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
        cv2.putText(frame, best_match[0], (x, y-10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

# Main Application
def main():
    load_face_database()
    ip_info = get_ip_details()
    cap = cv2.VideoCapture(0)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        recognize_faces(frame)

        # Get current status with thread safety
        with face_lock:
            face = current_face

        # Display information
        cv2.putText(frame, f"Face: {face}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        cv2.putText(frame, f"IP: {ip_info.get('query', 'Unknown')}", (10, frame.shape[0]-50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 100, 255), 2)  
        cv2.putText(frame, f"Location: {ip_info.get('country', 'Unknown')}", (10, frame.shape[0]-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 100, 255), 2) 

        cv2.imshow('Facial Recognition System', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Cleanup
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
