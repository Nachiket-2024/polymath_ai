import cv2
import os
import numpy as np
import time
import requests
from keras_facenet import FaceNet
from sklearn.metrics.pairwise import cosine_similarity
from concurrent.futures import ThreadPoolExecutor

# Suppress TensorFlow logs for faster execution
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Lazy load FaceNet
embedder = None

def get_embedder():
    global embedder
    if embedder is None:
        embedder = FaceNet()
    return embedder

# Face database
DATABASE_PATH = "database_images"
face_db = {}

# Load OpenCV cascade
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

def process_image(img_path):
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    faces = get_embedder().extract(img, threshold=0.95)
    return faces[0]['embedding'] if faces else None

# Load face embeddings from database using multithreading
def load_face_database():
    global face_db
    print("Loading face database...")
    
    with ThreadPoolExecutor() as executor:
        for person_name in os.listdir(DATABASE_PATH):
            person_folder = os.path.join(DATABASE_PATH, person_name)
            if os.path.isdir(person_folder):
                img_paths = [os.path.join(person_folder, img) for img in os.listdir(person_folder)]
                embeddings = list(executor.map(process_image, img_paths))
                embeddings = [e for e in embeddings if e is not None]
                
                if embeddings:
                    face_db[person_name] = np.mean(embeddings, axis=0)
    print("Face database loaded.")

load_face_database()

# Function to get IP details
def get_ip_details():
    """Fetches public IP details."""
    try:
        ip = requests.get("https://api64.ipify.org?format=json").json()["ip"]
        url = f"http://ip-api.com/json/{ip}"
        response = requests.get(url)
        if response.status_code == 200:
            return response.json()
    except Exception as e:
        return {"error": str(e)}
    return {"error": "Failed to retrieve data"}

# Get and print IP details
ip_details = get_ip_details()
print("IP Details:", ip_details)

# Initialize webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    faces = get_embedder().extract(rgb_frame, threshold=0.95)
    
    for face in faces:
        embedding = face['embedding']
        name = "Unknown"
        max_similarity = 0.5
        
        for person, db_embedding in face_db.items():
            similarity = cosine_similarity([embedding], [db_embedding])[0][0]
            if similarity > max_similarity:
                max_similarity = similarity
                name = person
        
        x, y, w, h = face['box']
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(frame, name, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    
    cv2.imshow('Face Recognition', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
