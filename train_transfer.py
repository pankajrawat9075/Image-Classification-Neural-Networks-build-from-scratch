import os
import argparse
from PIL import Image
import numpy as np
#import tensorflow as tf
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.utils.data import random_split
import torch
import wandb
wandb.login(key='e595ff5b95c353a42c4bd1f35b70856d4309ef00')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# data directory
data_dir = r'C:\Users\amitt\amit education\Mtech 2nd sem\DL\assignment2\nature_12K\inaturalist_12K'


params={}
if __name__=="__main__":
  parser = argparse.ArgumentParser(description = 'Input Hyperparameters')
  parser.add_argument('-wp'   , '--wandb_project'  , type = str  , default = 'CS22M010', metavar = '')
  parser.add_argument('-we'   , '--wandb_entity'   , type = str  , default = 'CS22M010', metavar = '')
  parser.add_argument('-e'   , '--epochs'   , type = int  , default = 10)
  params = vars(parser.parse_args())
num_epochs=params['epochs']
wandb.init(project=params['wandb_project'])

def get_transforms(data_augmentation):
    if data_augmentation=="yes":
        transform = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomRotation(degrees=30),
            transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
            transforms.Resize((224,224)),
            transforms.ToTensor()
        ])
    else:
        transform = transforms.Compose([
            transforms.Resize((224,224)),
            transforms.ToTensor()
        ])
    return transform

transform = transforms.Compose([transforms.Resize((224,224)),transforms.ToTensor()])
train_dataset=torchvision.datasets.ImageFolder(root=data_dir+'/train',transform=get_transforms("no"))
test_dataset=torchvision.datasets.ImageFolder(root=data_dir+'/val',transform=transform)
val_size = int(len(train_dataset) * 0.2)
train_size = len(train_dataset) - val_size
train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size])
train_loader=DataLoader(train_dataset,batch_size=16,shuffle=True)
val_loader=DataLoader(val_dataset,batch_size=16,shuffle=False)
test_loader=DataLoader(test_dataset,batch_size=16,shuffle=False)


import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.datasets as datasets # added for googlenet
#from torchvision.models import GoogLeNet
import torchvision.models as models


# class implementation for googlenet 
class MyGoogLeNet(nn.Module):
    def __init__(self, num_classes=10):
        super(MyGoogLeNet, self).__init__()
        self.googlenet = models.googlenet(pretrained=True)
        self.googlenet.fc = nn.Linear(self.googlenet.fc.in_features, num_classes)

        # Freeze all layers except the last fully connected layer
        for param in self.googlenet.parameters():
            param.requires_grad = False
        for param in self.googlenet.fc.parameters():
            param.requires_grad = True
            
    #forward pass is implemented      
    def forward(self, x):
        x = self.googlenet(x)
        return x

model = MyGoogLeNet().to(device)

# Define the loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=0.001, momentum=0.9)

# Train the model
for epoch in range(num_epochs):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)# Move data and target to the device
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

    # Evaluate the model on train_loader
    model.eval()
    train_correct = 0
    train_total = 0
    train_loss = 0.0
    with torch.no_grad():
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)# Move data and target to the device
            output = model(data)
            loss = criterion(output, target)
            train_loss += loss.item()
            _, predicted = torch.max(output.data, 1)
            train_total += target.size(0)
            train_correct += (predicted == target).sum().item()

    train_accuracy = 100 * train_correct/train_total
    train_loss /= len(train_loader)

    # Set model to evaluation mode
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(val_loader):
            data, target = data.to(device), target.to(device)# Move data and target to the device
            output = model(data)
            loss = criterion(output, target)
            val_loss += loss.item()
            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()

    # Calculate validation metrics
    val_loss /= len(val_loader)
    val_accuracy = 100 * correct / total

    # Print metrics for current epoch
    #print('Epoch: {} \t Training Loss: {:.6f}
    print("epoch",epoch)
    print('Training Loss: {:.6f} \t Training Accuracy: {:.6f}'.format(train_loss, train_accuracy))
    print('Validation Loss: {:.6f} \t Validation Accuracy: {:.6f}'.format(val_loss, val_accuracy))
    wandb.log({'train loss':train_loss,'train accuracy':train_accuracy,'valid loss':val_loss,'valid accuracy':val_accuracy})
