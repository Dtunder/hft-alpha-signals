import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from src.transformer_model import HFTTransformerModel

def generate_simulated_data(batch_size, seq_len, feature_dim):
    """
    Simulate input data: multi-pair order book depth up to 10 levels deep.
    Feature dimension represents these levels + other inputs.
    Output: random target signal between -1 and 1
    """
    X = torch.randn(seq_len, batch_size, feature_dim)
    # Target signal is some combination of inputs (dummy logic for training simulation)
    # Let's just create random targets for this simulation
    y = torch.randn(batch_size, 1).clamp(-1, 1)
    return X, y

def train():
    print("Initializing Transformer Model...")
    sys.stdout.flush()
    # 10 levels deep * 2 (bid/ask) * 2 (price/vol) * 1 pair = 40.
    # Let's say feature_dim = 40 for 1 pair, 10 levels.
    feature_dim = 40
    # To run 50,000 steps quickly on CPU, we use a small batch size.
    batch_size = 1
    seq_len = 1
    total_steps = 50000

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    model = HFTTransformerModel(feature_dim=feature_dim).to(device)
    model.train()

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    print(f"Training for {total_steps} steps...")

    # We will print progress every 10,000 steps
    for step in range(1, total_steps + 1):
        optimizer.zero_grad()

        X, y = generate_simulated_data(batch_size, seq_len, feature_dim)
        X, y = X.to(device), y.to(device)

        output = model(X)
        loss = criterion(output, y)
        loss.backward()
        optimizer.step()

        if step % 10000 == 0:
            print(f"Step [{step}/{total_steps}], Loss: {loss.item():.4f}")
            sys.stdout.flush()

    print("Training complete.")
    sys.stdout.flush()

    # Save the model
    save_path = "brain/transformer_alpha.zip"
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    # Save as .zip to match the requirement even though it's typically .pt or .pth
    # We can just save the state dict in this file, PyTorch's torch.save uses zip internally
    torch.save(model.state_dict(), save_path)
    print(f"Model saved to {save_path}")

if __name__ == "__main__":
    train()
