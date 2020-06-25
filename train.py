# Imports here
import argparse
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt 
import json
import time
import random, os
import torch
import torchvision 
import PIL 
from PIL import Image
from torchvision import datasets
from torchvision import transforms
from torchvision import models
from torch import nn
from torch import optim 
import torch.nn.functional as fun
from torch.autograd import Variable
import collections 
from collections import OrderedDict
from torch.optim import lr_scheduler
import numpy as np
from PIL import Image
import requests
import io
from io import BytesIO
import os
import copy
print("ok")


def get_input_args():
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('data_dir', help='Data Storage Directory')
    parser.add_argument('--save_dir', default='checkpoint.pth',
                        help='Check Point Directory')
    parser.add_argument('--arch',
                        type=str,
                        default='vgg16',
                        choices=['vgg16', 'densenet161'],
                        help='Network architecture')
    parser.add_argument('--hidden_units', default=512, type=int,
                        help='Number of Hidden Units')
    parser.add_argument('--epochs', default=1, type=int,
                        help='Number of epochs')
    parser.add_argument('--learning_rate', default=0.001, type=float,
                        help='Check Point Directory')
    parser.add_argument('--gpu',
                        type=bool,
                        nargs='?',
                        default=False,
                        const=True,
                        help='Use GPU for training')
    
    
 
    args = parser.parse_args()
    return args

def get_model(model_name):
    # implement two different model 
    if model_name == 'densenet161':
        model = models.densenet161(pretrained=True)
        
        classifier = nn.Sequential(nn.Linear(in_features=2208, out_features=256, bias=True),
                           nn.ReLU(),
                           nn.Dropout(p=0.4, inplace=False),
                           nn.Linear(in_features=256, out_features=133, bias=True),
                           nn.LogSoftmax(dim=1))
        
    elif model_name == 'vgg16':
        ###Load a pre-trained network which is the VGG13
        model = models.vgg16(pretrained=True)

        ###Defining  an untrained feed-forward network as a classifier
        classifier = nn.Sequential(
            nn.Linear(25088, 1024),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(1024, hidden_units),
            nn.ReLU(),
            nn.Dropout(p=0.2),    
            nn.Linear(hidden_units, 256),
            nn.ReLU(),
            nn.Linear(256, 102),
            nn.LogSoftmax(dim=1)
        )

    
        ###Freezing the  model
    for param in model.parameters():
        param.requires_grad = False

    model.classifier = classifier
    return model

args = get_input_args()
print(args)
hidden_units = args.hidden_units
epochs = args.epochs
learning_rate = args.learning_rate
save_dir = args.save_dir
is_gpu = args.gpu
arch = args.arch
print("hidden_units::: ", type(hidden_units), " :learning_rate ", type(learning_rate))

# Define the device
device = "cuda:0" if is_gpu else "cpu"
#device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("device:::: ", device)

data_dir = args.data_dir
train_dir = data_dir + '/train'
valid_dir = data_dir + '/valid'
test_dir = data_dir + '/test'


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


input_shape = 224 
batch_size = 42
scale = 256

data_transforms = {
    'train': transforms.Compose([transforms.RandomRotation(degrees=30),
                                 transforms.RandomResizedCrop(input_shape),
                                 transforms.RandomVerticalFlip(p=0.3),
                                 transforms.RandomHorizontalFlip(p=0.4),
                                 transforms.ToTensor(),
                                 transforms.Normalize([0.485, 0.456, 0.406],
                                                      [0.229, 0.224, 0.225])
    ]),
    'valid': transforms.Compose([transforms.Resize(scale),
                                 transforms.CenterCrop(input_shape),
                                 transforms.ToTensor(),
                                 transforms.Normalize([0.485, 0.456, 0.406],
                                                     [0.229, 0.224, 0.225])
        
    ]),
    'test': transforms.Compose([ transforms.Resize(scale),
                                 transforms.CenterCrop(input_shape),
                                 transforms.ToTensor(),
                                 transforms.Normalize([0.485, 0.456, 0.406],
                                                      [0.229, 0.224, 0.225])
       
    ])
}

# TODO: Load the datasets with ImageFolder
image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x]) for x in ['train', 'valid', 'test']}

# TODO: Using the image datasets and the trainforms, define the dataloaders
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], 
                                              batch_size=batch_size, 
                                              shuffle=True) for x in ['train', 'valid', 'test']}

dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'valid', 'test']}


print("the sizes of data:",dataset_sizes)


model = get_model(arch)
model = model.to(device)
print("our model::: ", model)

# criterion combines nn.LogSoftmax() and nn.NLLLoss() 
criterion = nn.CrossEntropyLoss()

# Implements stochastic gradient descent
optimizer_ft = optim.SGD(model.classifier.parameters(), lr=learning_rate, momentum=0.9)

# Decays the learning rate of each parameter group by gamma every step_size epochs
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)


def train_model(model, criterion, optimizer, scheduler, num_epochs=epochs):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch + 1, num_epochs))
        print('**' * 10)

        # train and valid each epoch 
        for phase in ['train', 'valid']:
            if phase == 'train':
                scheduler.step()
                model.train()  
            else:
                model.eval()   

            running_loss = 0.0
            running_corrects = 0

            # Iterations:
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()
                
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    if type(outputs) == tuple:
                        outputs, _ = outputs
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    #  in training backward + optimize 
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # running loss and corrects
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'valid' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model

print("done")

### Application of the model  
model = train_model(model, criterion, optimizer_ft, exp_lr_scheduler, num_epochs=epochs)

# Testing the performance and the accuracy of the network:
# Evaluation
print("acccc")
model.eval()

accuracy = 0

for inputs, labels in dataloaders['test']:
    inputs, labels = inputs.to(device), labels.to(device)
    outputs = model(inputs)
    
    # Class with the highest probability is our predicted class
    equality = (labels.data == outputs.max(1)[1])

    # Accuracy is number of correct predictions divided by all predictions
    accuracy += equality.type_as(torch.FloatTensor()).mean()
    
print("Test accuracy: {:.3f}".format(accuracy/len(dataloaders['test'])))
print("done")

## Save the checkpoint

checkpoint = {
    'model': model,
    'state_dict': model.state_dict(), 
    'optimizer': optimizer_ft.state_dict(), 
    'scheduler': exp_lr_scheduler.state_dict(),
    'class_to_idx': image_datasets['train'].class_to_idx,
    'epochs' :epochs
}

torch.save(checkpoint, save_dir)

print("done")