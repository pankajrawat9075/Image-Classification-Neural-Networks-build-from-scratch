
import argparse
import os
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

data_dir = r'C:\Users\amitt\amit education\Mtech 2nd sem\DL\assignment2\nature_12K\inaturalist_12K'

params={}
if __name__=="__main__":
  parser = argparse.ArgumentParser(description = 'Input Hyperparameters')
  parser.add_argument('-wp'   , '--wandb_project'  , type = str  , default = 'CS22M010', metavar = '')
  parser.add_argument('-we'   , '--wandb_entity'   , type = str  , default = 'CS22M010', metavar = '')
  parser.add_argument('-nnf'   , '--num_neuron_fc'   , type = int  , default = 10)
  parser.add_argument('-d'   , '--dropout'   , type = int  , default = 0.3)
  #parser.add_argument('-bn'   , '--batch_norm'   , type = int  , default = 1)
  parser.add_argument('-fso'   , '--filter_size_org'   , type = str  , default = "double")
  parser.add_argument('-fo'   , '--filter_org'   , type = str  , default = "same")
  parser.add_argument('-af'   , '--activation_fn'   , type = str  , default = "ReLU")
  parser.add_argument('-fs'   , '--filter_size'   , type = int  , default = 5)
  parser.add_argument('-nf'   , '--num_filters'   , type = int  , default = 96)
  params = vars(parser.parse_args())
#epochs=params['epochs']
num_filters = params['num_filters']
filter_size = params['filter_size']
activation_fn = params['activation_fn']
filter_org = params['filter_org']
filter_size_org=params['filter_size_org'] # extra added
batch_norm = False
dropout = params['dropout']
num_neuron_fc=params['num_neuron_fc']
wandb.init(project=params['wandb_project'])

 
#-------------------------------------------------------------------------------------------------------


def get_transforms(data_augmentation):
    if data_augmentation=="yes":
        transform = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomRotation(degrees=30),
            transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
            transforms.Resize((256,256)),
            transforms.ToTensor()
        ])
    else:
        transform = transforms.Compose([
            transforms.Resize((256,256)),
            transforms.ToTensor()
        ])
    return transform

transform = transforms.Compose([transforms.Resize((256,256)),transforms.ToTensor()])
train_dataset=torchvision.datasets.ImageFolder(root=data_dir+'/train',transform=get_transforms("yes"))
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
class Mish(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        return x * torch.tanh(F.softplus(x))
# Define the CNN model architecture
class CNN(nn.Module):
    def __init__(self, num_filters, filter_size, activation_fn, filter_org, batch_norm, dropout,num_neuron_fc,filter_size_org):
        super(CNN, self).__init__()
        if(activation_fn=="ReLU"):
            activation_fn=nn.ReLU
        if(activation_fn=="GELU"):
            activation_fn=nn.GELU
        if(activation_fn=="SiLU"):
            activation_fn=nn.SiLU
        if(activation_fn=="Mish"):
            activation_fn=Mish
        if(filter_org=="double"):
            filter_org=[1,2,2,2,2]
        if(filter_org=="same"):
            filter_org=[1,1,1,1,1]
        if(filter_org=="half"):
            filter_org=[1,0.5,0.5,0.5,0.5]
        if(filter_size_org=='same'):
            filter_size_org=[1,1,1,1,1]
        if(filter_size_org=="double"):
            filter_size_org=[1,2,2,2,2]
        if(filter_size_org=="half"):
            filter_size_org=[1,0.5,0.5,0.5,0.5]
        layers = []
        in_channels = 3
        w=256
        for i, f in enumerate(filter_org):
            out_channels = int(num_filters * f)
            filter_size1 = int(filter_size*filter_size_org[i])
            #calculate feature map dimension
            w=int((w-filter_size1+(2*1))+1)
            w=int(((w-2)//2)+1)
            #ends
            layers.append(nn.Conv2d(int(in_channels), int(out_channels), int(filter_size1), padding=1))
            if batch_norm:
                layers.append(nn.BatchNorm2d(out_channels))
            layers.append(activation_fn())
            layers.append(nn.MaxPool2d(2))
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            in_channels = out_channels
        #print(w)
        self.cnn = nn.Sequential(*layers)      
        self.fc1 = nn.Linear(int(out_channels) * w * w, num_neuron_fc)
        self.fc2 = nn.Linear(num_neuron_fc, 10)

    def forward(self, x):
        x = self.cnn(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.fc2(x)
        return x


# Initialize the model
model = CNN(num_filters, filter_size, activation_fn, filter_org, batch_norm, dropout, num_neuron_fc,filter_size_org).to(device)

# Define the loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Train the model
num_epochs = 1
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
    
