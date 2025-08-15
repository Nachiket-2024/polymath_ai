# ---------------------------- External Imports ----------------------------
import torch
import cv2
from torch.utils.data import TensorDataset, DataLoader

# ---------------------------- Internal Imports ----------------------------
from ..ai_core.learning.self_evolving_ai import SelfEvolvingAI
from ..live_facial_recognition import merge

# ---------------------------- Adaptive Integration Class ----------------------------
class FaceAIIntegration:
    """
    Integrates face recognition from merge.py with SelfEvolvingAI.
    Continuously adapts the AI model based on detected faces.
    """

    def __init__(self, device='cpu', dopamine_threshold=0.6, lr=0.001, fast_steps=5, task_batch_size=1):
        """
        Initialize integration.
        Args:
            device (str): 'cpu' or 'cuda'
            dopamine_threshold (float): Threshold below which AI adapts
            lr (float): Learning rate for meta-learner
            fast_steps (int): Number of quick adaptation steps
            task_batch_size (int): Batch size for adaptation
        """
        self.device = device
        self.task_batch_size = task_batch_size

        # Initialize SelfEvolvingAI
        self.ai_model = SelfEvolvingAI(device=device,
                                       dopamine_threshold=dopamine_threshold,
                                       lr=lr,
                                       fast_steps=fast_steps).to(device)

        # Load face database
        merge.load_face_database()

    def create_task_loader(self, embedding):
        """
        Creates a DataLoader for meta-learning adaptation using the given embedding.
        Args:
            embedding: torch.Tensor of shape [1, embedding_dim]
        Returns:
            DataLoader
        """
        # Here targets are dummy (zeros) as our reward system handles adaptation
        dataset = TensorDataset(embedding, torch.zeros(embedding.size(0), dtype=torch.long))
        loader = DataLoader(dataset, batch_size=self.task_batch_size)
        return loader

    def process_frame(self, frame):
        """
        Processes a single frame: detects faces, updates AI, adapts if dopamine low.
        Args:
            frame: np.ndarray from webcam
        Returns:
            tuple: (current face name, dopamine signal)
        """
        # Detect faces and update merge.current_face
        merge.recognize_faces(frame)
        with merge.face_lock:
            face_name = merge.current_face

        # Retrieve embedding
        with merge.face_lock:
            if face_name in merge.face_db:
                face_embedding = torch.tensor(merge.face_db[face_name], dtype=torch.float32).unsqueeze(0).to(self.device)
            else:
                face_embedding = torch.zeros(1, 512).to(self.device)

        # Create a task loader for adaptation
        task_loader = self.create_task_loader(face_embedding)

        # Forward pass through SelfEvolvingAI with task_loader for adaptation
        embeddings, dopamine = self.ai_model(face_embedding, task_loader=task_loader)

        return face_name, dopamine

    def run(self):
        """
        Run webcam loop integrating face recognition and AI updates.
        """
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Error: Webcam not detected.")
            return

        print("Starting adaptive integration loop. Press 'q' to quit.")
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            face_name, dopamine = self.process_frame(frame)

            # Display information
            cv2.putText(frame, f"Face: {face_name}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
            cv2.putText(frame, f"Dopamine: {dopamine:.2f}", (10, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

            cv2.imshow("Adaptive Face-AI Integration", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

# ---------------------------- Main Entry ----------------------------
if __name__ == "__main__":
    integrator = FaceAIIntegration(device='cpu')
    integrator.run()
