#The training script trains the model and saves it to the checkpoint file.
#Imports
import torch
from torch import nn
import seaborn as sns
from collections import OrderedDict
import argparse
from torch import optim
import torchvision
import torch.nn.functional as F
from torchvision import datasets, transforms, models
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from PIL import Image
import numpy as np
import json

parser = argparse.ArgumentParser (description = "Training script")
parser.add_argument ('--arch', help = 'Vgg16 can be used if provided with valid argument else densenet121 is used', type = str)
parser.add_argument ('data_dir', help = 'Please Provide data directory(mandatory)', type = str)
parser.add_argument ('--checkpoint_dir', help = 'Provide saving directory. (Optional)', type = str)
parser.add_argument ('--lr', help = 'Learning rate, default value 0.001', type = float)
parser.add_argument ('--hidden_nodes', help = 'Number hidden nodes in Classifier. Default value is 256', type = int)
parser.add_argument ('--epochs',help = 'Number of epochs...Default 15', type = int)
parser.add_argument ('--GPU', help = "Option to use GPU", type = str)
args = parser.parse_args ()


data_dir = args.data_dir
train_dir = data_dir + '/train'
valid_dir = data_dir + '/valid'
test_dir = data_dir + '/test'
device='cuda' if args.GPU=='GPU' else 'cpu'

if data_dir:
    train_transforms = transforms.Compose([transforms.RandomRotation(45),
                                      transforms.RandomResizedCrop(224),
                                      transforms.RandomHorizontalFlip(),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485,0.456,0.406],
                                                          [0.229,0.224,0.225])])
    valid_transforms=transforms.Compose([transforms.Resize(255),
                                    transforms.CenterCrop(224),
                                    transforms.ToTensor(),
                                    transforms.Normalize([0.485,0.456,0.406],
                                                          [0.229,0.224,0.225])])

    test_transforms=transforms.Compose([transforms.Resize(255),
                                    transforms.CenterCrop(224),
                                    transforms.ToTensor(),
                                   transforms.Normalize([0.485,0.456,0.406],
                                                          [0.229,0.224,0.225])])

    train_datasets = datasets.ImageFolder(train_dir,transform=train_transforms)
    valid_datasets = datasets.ImageFolder(valid_dir,transform=valid_transforms)
    test_datasets  = datasets.ImageFolder(test_dir,transform=test_transforms)

    
    trainloaders = torch.utils.data.DataLoader(train_datasets,batch_size=64,shuffle=True)
    validloaders = torch.utils.data.DataLoader(valid_datasets,batch_size=64,shuffle=True)
    testloaders  = torch.utils.data.DataLoader(test_datasets,batch_size=64) 
    
with open('cat_to_name.json', 'r') as f:
    cat_to_name = json.load(f)
   
#Funktion1
def build_model(arch,hidden_nodes):
    
    if arch=='vgg16':
    
        model=models.vgg16(pretrained=True)
        for param in model.parameters():
            param.requires_grad = False
        
        if hidden_nodes:

                    new_classifier=nn.Sequential(OrderedDict([
                                    ('fc1', nn.Linear(25088,512)),
                                    ('relu1',nn.ReLU()),
                                    ('dropout1', nn.Dropout(0.2)),
                                    ('fc2',nn.Linear(512,hidden_nodes)),
                                    ('relu2',nn.ReLU()),
                                    ('dropout2',nn.Dropout(0.2)),
                                    ('fc3',nn.Linear(hidden_nodes,102)),
                                    ('output',nn.LogSoftmax(dim=1))]))

        else:
                      new_classifier=nn.Sequential(OrderedDict([
                                    ('fc1', nn.Linear(25088,512)),
                                    ('relu1',nn.ReLU()),
                                    ('dropout1', nn.Dropout(0.2)),
                                    ('fc2',nn.Linear(512,256)),
                                    ('relu2',nn.ReLU()),
                                    ('dropout2',nn.Dropout(0.2)),
                                    ('fc3',nn.Linear(256,102)),
                                    ('output',nn.LogSoftmax(dim=1))]))
    
    
    else:
        
        arch='densenet121'
    
        model=models.densenet121(pretrained=True)
        for param in model.parameters():
            param.requires_grad = False
        
        if hidden_nodes:

                    new_classifier=nn.Sequential(OrderedDict([
                                    ('fc1', nn.Linear(1024,512)),
                                    ('relu1',nn.ReLU()),
                                    ('dropout1', nn.Dropout(0.2)),
                                    ('fc2',nn.Linear(512,hidden_nodes)),
                                    ('relu2',nn.ReLU()),
                                    ('dropout2',nn.Dropout(0.2)),
                                    ('fc3',nn.Linear(hidden_nodes,102)),
                                    ('output',nn.LogSoftmax(dim=1))]))

        else:
                      new_classifier=nn.Sequential(OrderedDict([
                                    ('fc1', nn.Linear(1024,512)),
                                    ('relu1',nn.ReLU()),
                                    ('dropout1', nn.Dropout(0.2)),
                                    ('fc2',nn.Linear(512,256)),
                                    ('relu2',nn.ReLU()),
                                    ('dropout2',nn.Dropout(0.2)),
                                    ('fc3',nn.Linear(256,102)),
                                    ('output',nn.LogSoftmax(dim=1))]))
    model.classifier=new_classifier
    return model,arch

model,architecture= build_model (args.arch,args.hidden_nodes)

criterion=nn.NLLLoss()
if args.lr:
    optimizer=optim.Adam(model.classifier.parameters(),lr=args.lr)
else:
    optimizer=optim.Adam(model.classifier.parameters(),lr=0.001)

model.to(device);

if args.epochs:
    epochs=args.epochs
else:
    epochs=2
print("TRAINING STARTED---------------------------")    
steps = 0
running_loss = 0
print_every = 5
train_losses,validation_losses=[],[]
    
for epoch in range(epochs):
    for images, labels in trainloaders:
        steps += 1
        images, labels = images.to(device), labels.to(device)
        print(steps)
        optimizer.zero_grad()

        log_loss = model.forward(images)
        loss = criterion(log_loss, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()

        if steps % print_every == 0:
            valid_loss = 0
            accuracy = 0
            model.eval()
            with torch.no_grad():
                for images, labels in validloaders:
                    images, labels = images.to(device), labels.to(device)
                    log_loss = model.forward(images)
                    valid_loss += criterion(log_loss, labels)

                    ps = torch.exp(log_loss)
                    top_p, top_class = ps.topk(1, dim=1)
                    equals = top_class == labels.view(*top_class.shape)
                    accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

            train_losses.append (running_loss/len(trainloaders))
            validation_losses.append(valid_loss/len(validloaders))

            print(f"Epoch {epoch+1}/{epochs}.. "
                  f"Training loss: {running_loss/print_every:.3f}.. "
                  f"Validation loss: {valid_loss/len(validloaders):.3f}.. "
                  f"Validation accuracy: {(accuracy/len(validloaders))*100:.3f}")
            running_loss = 0
            model.train()
print("Training over")
def testing():
    accuracy=0

    model.to(device)

    model.eval()

    with torch.no_grad():

        for images,labels in testloaders:

            images,labels=images.to(device),labels.to(device)

            optimizer.zero_grad()

            log_loss=model.forward(images)

            ps=torch.exp(log_loss)

            top_prob,top_class=ps.topk(1,dim=1)

            equals= top_class == labels.view(*top_class.shape)

            accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

        model.train()
        
    return (accuracy/len(testloaders))*100
    
model.class_to_idx = train_datasets.class_to_idx
checkpoint = {'class_to_idx': model.class_to_idx,
              
              'classifier':model.classifier,
              
              'arch': architecture,
              
              'state_dict': model.state_dict()}
if args.checkpoint_dir:
    torch.save (checkpoint, args.checkpoint_dir + '/checkpoint_command.pth')
else:
    torch.save (checkpoint, 'checkpoint_command.pth')