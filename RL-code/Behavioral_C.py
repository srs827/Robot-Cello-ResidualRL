import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from torch.utils.data import Dataset, DataLoader

class TrajectoryDataset(Dataset):
    def __init__(self, csv_path):
        df = pd.read_csv(csv_path)
        print("Loaded columns:", df.columns.tolist())  
        joint_cols = ['q_base', 'q_shoulder', 'q_elbow', 'q_wrist1', 'q_wrist2', 'q_wrist3']

        if not all(col in df.columns for col in joint_cols):
            raise ValueError("Missing joint columns")

        self.states = df[joint_cols].values[:-1]
        self.next_states = df[joint_cols].values[1:]
        self.actions = self.next_states - self.states


    def __len__(self):
        return len(self.states)

    def __getitem__(self, idx):
        return (
            torch.tensor(self.states[idx], dtype=torch.float32),
            torch.tensor(self.actions[idx], dtype=torch.float32),
        )

class PolicyNet(nn.Module):
    def __init__(self, input_dim=6, output_dim=6, hidden_size=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_dim)
        )

    def forward(self, x):
        return self.net(x)

def train_bc(csv_path, output_path="bc_policy_pretrained.pt", epochs=100, batch_size=64, lr=1e-3):
    dataset = TrajectoryDataset(csv_path)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = PolicyNet()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    for epoch in range(epochs):
        total_loss = 0.0
        for state, action in loader:
            pred = model(state)
            loss = criterion(pred, action)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch {epoch + 1}/{epochs} | Loss: {total_loss:.6f}")

    torch.save(model.state_dict(), output_path)
    print(f"Behavior cloning model saved to: {output_path}")

if __name__ == "__main__":
    train_bc("allegro-log-detailed.csv")
