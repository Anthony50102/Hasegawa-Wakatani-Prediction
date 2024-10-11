import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm

class DerivedCNN(nn.Module):
    def __init__(self):
        super(DerivedCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(64 * 64 * 64, 128)
        self.fc2 = nn.Linear(128, 1)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(-1, 64 * 64 * 64) # Reshape for Relu
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x.squeeze() # Remove singleton dims

# Training function
def train_one_epoch(model, train_loader, criterion, optimizer, device, batchsize):
    model.train()
    running_loss = 0.0
    for batch in tqdm(train_loader, desc="Training",total=batchsize):
        inputs = batch['x']
        labels = batch['y']
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    return running_loss / len(train_loader)

# Evaluation function
def evaluate(model, val_loader, criterion, device):
    model.eval()
    running_loss = 0.0
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Evaluating"):
            inputs = batch['x']
            labels = batch['y']
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            running_loss += loss.item()
    return running_loss / len(val_loader)

def train(model, 
          train_loader, 
          test_loader, 
          num_epochs = 20, 
          batch_size = 32, 
          learning_rate = 0.001,
          criterion = nn .MSELoss(), 
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    for epoch in range(num_epochs):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device, batch_size)
        val_loss = evaluate(model, test_loader, criterion, device)
        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

    print("Training completed!")