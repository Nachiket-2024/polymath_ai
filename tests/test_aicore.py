import torch
from transformers import AutoTokenizer
from torch_geometric.data import Data  # If you use torch_geometric for graphs

from ..ai_core.ai_core import AICore

def test_aicore():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Initialize model
    model = AICore(device=device).to(device)
    model.eval()  # inference mode

    # Tokenize example text input
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    sample_text = ["this is a test"]
    encoded_input = tokenizer(sample_text, padding=True, truncation=True, return_tensors="pt").to(device)

    # Dummy audio and vision inputs (assuming batch size 1, adapt as needed)
    audio_input = torch.randn(1, 128).to(device)  # dummy feature vector for audio
    vision_input = torch.randn(1, 3, 224, 224).to(device)  # dummy image tensor

    # Dummy graph data for GNN (example: 5 nodes, 3 features each)
    x = torch.randn(5, 512).to(device)  # node features matching your GNN input dim
    edge_index = torch.tensor([[0,1,2,3],[1,2,3,4]], dtype=torch.long).to(device)  # sample edges
    graph_data = Data(x=x, edge_index=edge_index)

    # Forward pass
    with torch.no_grad():
        embedding, dopamine = model(encoded_input, audio_input, vision_input, graph_data)

    print("Embedding shape:", embedding.shape)
    print("Dopamine signal:", dopamine.item())

if __name__ == "__main__":
    test_aicore()
