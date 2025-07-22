# import os
# import torch
# import torchvision.transforms as transforms
# import torchvision.datasets as datasets
# from torch.utils.data import random_split, DataLoader
# from torchvision import models
# from torchvision.models import ResNet18_Weights
# import torch.nn as nn
# import torch.optim as optim
# from tqdm import tqdm

# print("🚀 Starting training script...")

# # Data transforms with smaller resolution
# transform = transforms.Compose([
#     transforms.Resize((128, 128)),
#     transforms.ToTensor()
# ])

# # Dataset path
# data_dir = "D:/DISEASE_PRED/dataset/PlantVillage"
# print(f"📂 Loading dataset from: {data_dir}")
# dataset = datasets.ImageFolder(data_dir, transform=transform)
# class_names = dataset.classes
# print(f"✅ Found {len(class_names)} classes: {class_names}")
# print(f"🖼️ Total images: {len(dataset)}")

# # Split dataset
# train_size = int(0.8 * len(dataset))
# val_size = len(dataset) - train_size
# train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
# train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
# val_loader = DataLoader(val_dataset, batch_size=32)

# # Model setup
# print("🧠 Initializing ResNet18...")
# model = models.resnet18(weights=ResNet18_Weights.DEFAULT)
# model.fc = nn.Linear(model.fc.in_features, len(class_names))

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model = model.to(device)
# print(f"💻 Using device: {device}")

# criterion = nn.CrossEntropyLoss()
# optimizer = optim.Adam(model.parameters(), lr=0.001)

# # Training loop
# print("🏋️ Training started...")
# epochs = 5
# for epoch in range(epochs):
#     model.train()
#     running_loss = 0.0
#     for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
#         inputs, labels = inputs.to(device), labels.to(device)
#         optimizer.zero_grad()
#         outputs = model(inputs)
#         loss = criterion(outputs, labels)
#         loss.backward()
#         optimizer.step()
#         running_loss += loss.item()
    
#     avg_loss = running_loss / len(train_loader)
#     print(f"📊 Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")

# # Save model
# model_path = "../model/crop_disease_model.pt"
# os.makedirs("../model", exist_ok=True)
# torch.save({
#     "model": model.state_dict(),
#     "class_names": class_names
# }, model_path)
# print(f"💾 Model saved to: {model_path}")
# print("✅ Training complete.")


import os
import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import random_split, DataLoader
from torchvision import models
from torchvision.models import ResNet18_Weights
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

print("🚀 Starting training script...")

# ✅ Use proper normalization for pretrained ResNet
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# Dataset path
data_dir = "D:/DISEASE_PRED/dataset/PlantVillage"
print(f"📂 Loading dataset from: {data_dir}")
dataset = datasets.ImageFolder(data_dir, transform=transform)
class_names = dataset.classes
print(f"✅ Found {len(class_names)} classes: {class_names}")
print(f"🖼️ Total images: {len(dataset)}")

# Split dataset
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32)

# Model setup
print("🧠 Initializing ResNet18...")
model = models.resnet18(weights=ResNet18_Weights.DEFAULT)
model.fc = nn.Linear(model.fc.in_features, len(class_names))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
print(f"💻 Using device: {device}")

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
epochs = 1
print("🏋️ Training started...")
for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} - Training"):
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    avg_loss = running_loss / len(train_loader)
    print(f"📉 Epoch {epoch+1}, Training Loss: {avg_loss:.4f}")

    # ✅ Validation phase
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} - Validation"):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print(f"✅ Epoch {epoch+1}, Validation Accuracy: {accuracy:.2f}%")

# Save model
model_dir = "D:/DISEASE_PRED/model"
os.makedirs(model_dir, exist_ok=True)
model_path = os.path.join(model_dir, "crop_disease_model.pt")
torch.save({
    "model": model.state_dict(),
    "class_names": class_names
}, model_path)
print(f"💾 Model saved to: {model_path}")
print("✅ Training complete.")
