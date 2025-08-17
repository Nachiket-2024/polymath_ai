# ---------------------------- External Imports ----------------------------
import cv2  # For webcam capture and display
import torch  # For AI model inference
import numpy as np  # For numerical operations

# ---------------------------- Internal Imports ----------------------------
from facial_recognition.merge import embedder, load_face_database  # FaceNet-based face embedding system
from ai_core.learning.self_evolving_ai import SelfEvolvingAI  # Core AI model

# ---------------------------- Config ----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------------------- Initialize Model ----------------------------
ai_model = SelfEvolvingAI().to(device)
ai_model.eval()

# ---------------------------- Video Test Function ----------------------------
def test_video():
    # Load registered faces into memory (precomputes embeddings)
    load_face_database()

    # Open webcam stream
    cap = cv2.VideoCapture(0)
    print("Starting AI test with FaceNet embeddings. Press 'q' to quit.")

    while True:
        # Capture a frame from the webcam
        ret, frame = cap.read()
        if not ret:
            break

        # Convert BGR (OpenCV format) to RGB for embedding
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Extract faces + embeddings (FaceNet gives 512-d vectors)
        faces = embedder.extract(rgb_frame, threshold=0.90)

        if faces:
            # Take the first face embedding
            face_embedding = faces[0]["embedding"]
            
            # Convert to torch tensor, add batch dimension, and send to device
            face_tensor = torch.tensor(face_embedding, dtype=torch.float32).unsqueeze(0).to(device)

            # Forward pass through AI model (text_input=None, only embeddings)
            embeddings, dopamine = ai_model(text_input=None, vision_input=face_tensor)

            # Print outputs for debugging
            print("Embedding shape:", embeddings.shape)
            print("Dopamine signal:", dopamine)

        # Display the video feed
        cv2.imshow("AI Face Recognition", frame)

        # Exit if user presses 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release resources after loop ends
    cap.release()
    cv2.destroyAllWindows()


# ---------------------------- Main Entry ----------------------------
if __name__ == "__main__":
    test_video()
