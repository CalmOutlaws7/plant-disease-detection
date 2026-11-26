import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets,models,transforms
from torch.utils.data import DataLoader
import os
device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
transform=transforms.Compose([transforms.Resize((224,224)),transforms.RandomHorizontalFlip(),transforms.ToTensor(),transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])
dataset_path="./dataset"
dataset=datasets.ImageFolder(dataset_path,transform=transform)
class_names=dataset.classes
print(f"Classes found: {class_names}")
train_size=int(0.8*len(dataset))
val_size=len(dataset)-train_size
train_data,val_data=torch.utils.data.random_split(dataset,[train_size,val_size])
train_loader=DataLoader(train_data,batch_size=32,shuffle=True)
val_loader=DataLoader(val_data,batch_size=32,shuffle=False)
model=models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
model.classifier[1]=nn.Linear(model.last_channel,len(class_names))
model=model.to(device)
criterion=nn.CrossEntropyLoss()
optimizer=optim.Adam(model.parameters(),lr=0.001)
epochs=5
for epoch in range(epochs):
    model.train()
    running_loss=0.0
    for inputs,labels in train_loader:
        inputs,labels=inputs.to(device),labels.to(device)
        optimizer.zero_grad()
        outputs=model(inputs)
        loss=criterion(outputs,labels)
        loss.backward()
        optimizer.step()
        running_loss+=loss.item()*inputs.size(0)
    epoch_loss=running_loss/len(train_loader.dataset)
    print(f"Epoch [{epoch+1}/{epochs}], Loss: {epoch_loss:.4f}")
torch.save({'model_state':model.state_dict(),'classes':class_names},'plant_disease_model.pth')
print("Model trained and saved successfully as 'plant_disease_model.pth'!")
