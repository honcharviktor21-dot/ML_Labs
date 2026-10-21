import time
import numpy as np
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split
from torchinfo import summary
from thop import profile
import matplotlib.pyplot as plt

# Student ID: 72014

# ==========================================
# 1. Dataset Setup
# ==========================================
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.2860,), (0.3530,))
])

full_train = torchvision.datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)
train_set, val_set = random_split(full_train, [55000, 5000])
test_set = torchvision.datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform)

train_loader = DataLoader(train_set, batch_size=128, shuffle=True)
val_loader = DataLoader(val_set, batch_size=128, shuffle=False)
test_loader = DataLoader(test_set, batch_size=128, shuffle=False)

# ==========================================
# 2. Architectures
# ==========================================
class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(784, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )
    def forward(self, x):
        return self.net(x)

class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(32 * 7 * 7, 10)
        )
    def forward(self, x):
        return self.net(x)

class AttentionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.patch_proj = nn.Linear(49, 64)
        encoder_layer = nn.TransformerEncoderLayer(d_model=64, nhead=4, dim_feedforward=128, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.fc = nn.Linear(64, 10)

    def forward(self, x):
        batch_size = x.size(0)
        x = x.view(batch_size, 1, 4, 7, 4, 7)
        x = x.permute(0, 2, 4, 3, 5, 1).contiguous()
        x = x.view(batch_size, 16, 49)
        
        x = self.patch_proj(x)
        x = self.transformer(x)
        x = x.mean(dim=1)
        x = self.fc(x)
        return x

models = {
    'MLP': MLP().to(device),
    'CNN': CNN().to(device),
    'Attention': AttentionModel().to(device)
}

# ==========================================
# 3. Profiling & Latency Measurements
# ==========================================
results = {}

for name, model in models.items():
    print(f"\n=================== Profiling {name} ===================")
    dummy_input = torch.randn(1, 1, 28, 28).to(device)
    
    summary(model, input_size=(1, 1, 28, 28))
    
    macs, params = profile(model, inputs=(dummy_input,), verbose=False)
    flops_thop = macs * 2 
    print(f"THOP Params: {int(params):,}, THOP FLOPs: {int(flops_thop):,}")
    
    model.eval()
    latencies = {}
    for b_size, runs in [(1, 100), (128, 20)]:
        b_input = torch.randn(b_size, 1, 28, 28).to(device)
        with torch.no_grad():
            for _ in range(5): model(b_input)
            
            run_times = []
            for _ in range(runs):
                if device.type == 'cuda': torch.cuda.synchronize()
                t0 = time.perf_counter()
                model(b_input)
                if device.type == 'cuda': torch.cuda.synchronize()
                run_times.append((time.perf_counter() - t0) * 1000)
                
            median_latency = np.median(run_times)
            latencies[b_size] = median_latency
            print(f"Batch Size {b_size} Latency: {median_latency:.3f} ms (Median over {runs} runs)")
            
    results[name] = {
        'params': int(params),
        'flops': int(flops_thop),
        'lat_b1': latencies[1],
        'lat_b128': latencies[128]
    }

# ==========================================
# 4. Training Loop (5 Epochs)
# ==========================================
print("\n=================== Training Models ===================")
criterion = nn.CrossEntropyLoss()

for name, model in models.items():
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    print(f"Training {name} for 5 epochs...")
    
    for epoch in range(5):
        model.train()
        running_loss = 0.0
        for X, y in train_loader:
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            out = model(X)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f"  Epoch {epoch+1}/5 - Loss: {running_loss / len(train_loader):.4f}")
        
    model.eval()
    correct = 0
    with torch.no_grad():
        for X, y in test_loader:
            X, y = X.to(device), y.to(device)
            preds = model(X).argmax(dim=1)
            correct += (preds == y).sum().item()
            
    test_acc = (correct / len(test_set)) * 100
    results[name]['accuracy'] = test_acc
    print(f"--> {name} Final Test Accuracy: {test_acc:.2f}%")

# ==========================================
# 5. Plotting Accuracy vs Params and Latency
# ==========================================
names = list(results.keys())
accuracies = [results[m]['accuracy'] for m in names]
params_list = [results[m]['params'] for m in names]
latencies_128 = [results[m]['lat_b128'] for m in names]

plt.figure(figsize=(8, 5))
plt.scatter(params_list, accuracies, color='blue', s=100)
for i, name in enumerate(names):
    plt.annotate(f"{name} ({params_list[i]:,} params)", (params_list[i], accuracies[i]),
                 textcoords="offset points", xytext=(0,10), ha='center')
plt.xlabel("Parameter Count")
plt.ylabel("Test Accuracy (%)")
plt.title("Accuracy vs Parameter Count")
plt.grid(True)
plt.savefig("accuracy_vs_params.png")
plt.close()

plt.figure(figsize=(8, 5))
plt.scatter(latencies_128, accuracies, color='green', s=100)
for i, name in enumerate(names):
    plt.annotate(f"{name} ({latencies_128[i]:.2f} ms)", (latencies_128[i], accuracies[i]),
                 textcoords="offset points", xytext=(0,10), ha='center')
plt.xlabel("Inference Latency (Batch 128) [ms]")
plt.ylabel("Test Accuracy (%)")
plt.title("Accuracy vs Latency")
plt.grid(True)
plt.savefig("accuracy_vs_latency.png")
plt.close()

print("\nSaved plots: 'accuracy_vs_params.png' and 'accuracy_vs_latency.png'")