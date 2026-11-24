import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader, ConcatDataset

# 1. Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 2. Data Transformations 
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# 3. Load Datasets
data_dir = "./dataset"
subsets = []

# List of dataset folder names 
dataset_folders = [
    "Fig Leaves Dataset",
    "AJLCD-2025",
    "MangoLeafBD Dataset",
    "MoringaLeafNet A Multi-Class Leaf Disease Dataset",
    "Pudina Dataset"
]

for folder in dataset_folders:
    path = os.path.join(data_dir, folder)
    if os.path.exists(path):
        subsets.append(datasets.ImageFolder(path, transform=transform))

if not subsets:
    print("Error: No dataset folders found! Check your folder names.")
else:
    # Combine all plant datasets into one unified training pool
    combined_dataset = ConcatDataset(subsets)
    
    all_classes = sorted(list(set([cls for ds in subsets for cls in ds.classes])))
    print(f"Total Combined Classes ({len(all_classes)}): {all_classes}")

    train_loader = DataLoader(combined_dataset, batch_size=32, shuffle=True, num_workers=0)

    # 4. Load Pretrained MobileNetV2 optimized for speed and accuracy
    model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
    model.classifier[1] = nn.Linear(model.last_channel, len(all_classes))
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # 5. Fast Training Loop 
    epochs = 3
    print("Starting model training on your RTX 5060...")
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
        
        epoch_loss = running_loss / total
        epoch_acc = 100.0 * correct / total
        print(f"Epoch [{epoch+1}/{epochs}] | Loss: {epoch_loss:.4f} | Accuracy: {epoch_acc:.2f}%")

    torch.save({'model_state': model.state_dict(), 'classes': all_classes}, 'plant_disease_model.pth')
    print("Model successfully trained and saved as 'plant_disease_model.pth'!")