import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import joblib
import os

class GoalConditionedTCN(nn.Module):
    def __init__(self, input_size, num_channels, kernel_size=2, dropout=0.2):
        super(GoalConditionedTCN, self).__init__()
        # input_size should be number of pairs (e.g., 2) + 1 (for goal condition)

        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = input_size if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]

            # Conv1d expects (batch_size, channels, length)
            layers += [
                nn.Conv1d(in_channels, out_channels, kernel_size,
                          stride=1, dilation=dilation_size, padding=(kernel_size-1) * dilation_size),
                nn.ReLU(),
                nn.Dropout(dropout)
            ]

        self.network = nn.Sequential(*layers)
        # Output layers
        self.fc = nn.Linear(num_channels[-1], 3) # Output: 0=HOLD, 1=BUY, 2=SELL

    def forward(self, x):
        # x is (batch_size, sequence_length, input_size)
        # We need to transpose to (batch_size, input_size, sequence_length) for Conv1d
        x = x.transpose(1, 2)
        out = self.network(x)
        # Take the last output in the sequence
        out = out[:, :, -1]
        out = self.fc(out)
        return out

def generate_synthetic_data(num_samples=1000, seq_len=10, num_pairs=2):
    """
    Generates synthetic multi-pair OBI data and a goal condition.
    """
    X = []
    y = []

    for _ in range(num_samples):
        # Generate random OBIs for multiple pairs between -1 and 1
        obis = np.random.uniform(-1, 1, (seq_len, num_pairs))

        # Goal condition (e.g., target risk/return profile) between 0 and 1
        goal = np.random.uniform(0, 1, (seq_len, 1))

        # Combine OBIs and goal
        seq_data = np.concatenate([obis, goal], axis=1)
        X.append(seq_data)

        # Labeling logic:
        # If the average OBI of the last tick across pairs is > 0.5 -> BUY (1)
        # If < -0.5 -> SELL (2)
        # Else -> HOLD (0)
        last_obi_avg = np.mean(obis[-1])
        if last_obi_avg > 0.5:
            label = 1
        elif last_obi_avg < -0.5:
            label = 2
        else:
            label = 0
        y.append(label)

    return np.array(X, dtype=np.float32), np.array(y, dtype=np.int64)

def train_model():
    print("Generating synthetic multi-pair OBI data...")
    X_train, y_train = generate_synthetic_data(num_samples=2000)

    X_train_tensor = torch.tensor(X_train)
    y_train_tensor = torch.tensor(y_train)

    # Model configuration
    input_size = X_train.shape[2] # num_pairs + 1 (goal)
    num_channels = [16, 32, 64]

    model = GoalConditionedTCN(input_size, num_channels)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    epochs = 50
    batch_size = 32

    print("Training Goal-Conditioned TCN...")
    for epoch in range(epochs):
        permutation = torch.randperm(X_train_tensor.size()[0])

        epoch_loss = 0
        for i in range(0, X_train_tensor.size()[0], batch_size):
            indices = permutation[i:i+batch_size]
            batch_x, batch_y = X_train_tensor[indices], y_train_tensor[indices]

            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        if (epoch+1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{epochs}], Loss: {epoch_loss/len(X_train_tensor):.4f}")

    print("Training complete.")

    # Ensure brain directory exists
    os.makedirs('brain', exist_ok=True)

    # Save model using joblib to brain/multi_alpha_model.zip as requested
    model_state = model.state_dict()
    joblib.dump(model_state, 'brain/multi_alpha_model.zip')
    print("Model saved to brain/multi_alpha_model.zip")

if __name__ == "__main__":
    train_model()
