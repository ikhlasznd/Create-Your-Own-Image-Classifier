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

def get_input_args():
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('checkpoint', help='Loading Checkpoint')
    parser.add_argument('image_path', help='Image File Path')
    parser.add_argument('--gpu',
                        type=bool,
                        nargs='?',
                        default=False,
                        const=True,
                        help='Use GPU for training')
    parser.add_argument('--top_k', default=3, type=int,
                        help='Top K Categories')
    parser.add_argument('--category_names',
                        type=str,
                        default='cat_to_name.json',
                        help='Categories File')

 
    args = parser.parse_args()
    return args


args = get_input_args()
print(args)
checkpoint = args.checkpoint
is_gpu = args.gpu
image_path = args.image_path
top_k = args.top_k
category_names = args.category_names

device = "cuda:0" if is_gpu else "cpu"

with open(category_names, 'r') as f:
    cat_to_name = json.load(f)

# Loading the checkpoint and rebuilding the network

def load_checkpoint(filename):
    # Note: Input model & optimizer should be pre-defined.  This routine only updates their states.
    start_epoch = 0
    if os.path.isfile(filename):
        print("=> loading checkpoint '{}'".format(filename))
        checkpoint = torch.load(filename)
        start_epoch = checkpoint['epochs']
        
        model = checkpoint['model']
        model = model.to(device)
        model.class_to_idx = checkpoint['class_to_idx']
        model.load_state_dict(checkpoint['state_dict'])
        
        optimizer = optim.SGD(model.classifier.parameters(), lr=0.001, momentum=0.9)
        optimizer.load_state_dict(checkpoint['optimizer'])
        
        scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
        scheduler.load_state_dict(checkpoint['scheduler'])
        print("=> loaded checkpoint '{}' (epoch {})".format(filename, checkpoint['epochs']))
    else:
        print("=> no checkpoint found at '{}'".format(filename))

    return model, optimizer, start_epoch, scheduler


model_load, optimizer_load, start_epoch, scheduler_load = load_checkpoint(checkpoint)

model_load
print("done")

# Inference for classification

def process_image(image):
      
       # scale
    scale_size = 256,256
    image.thumbnail(scale_size, Image.LANCZOS)

    # crop
    crop_size = 224
    width, height = image.size   # Get dimensions
    
    left = (width - crop_size)/2
    top = (height - crop_size)/2
    right = (width + crop_size)/2
    bottom = (height + crop_size)/2
    
    image = image.crop((left, top, right, bottom))
    
    # normalize
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image_array = np.array(image) / 255
    
    image = (image_array - mean) / std
    
    # reorder dimensions
    image = image.transpose((2,0,1))
    
    return torch.from_numpy(image)

def imshow(image, ax=None, title=None):
    """Imshow for Tensor."""
    if ax is None:
        fig, ax = plt.subplots()
    
    # PyTorch tensors assume the color channel is the first dimension
    # but matplotlib assumes is the third dimension
    image = image.numpy().transpose((1, 2, 0))
    
    # Undo preprocessing
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = std * image + mean
    
    # Image needs to be clipped between 0 and 1 or it looks like noise when displayed
    image = np.clip(image, 0, 1)
    
    ax.imshow(image)
    
    return ax
print("done")


# Class Prediction

def predict(image_pil, model, top_k):
   
    image = process_image(image_pil)
    image = image.unsqueeze_(0)
    image = image.cuda().float()
    
    model.eval()
    
    with torch.no_grad():
        output = model(image)
        prob, idxs = torch.topk(output, top_k)
        #_, preds = torch.max(output.data, 1)
    
        # convert indices to classes
        idxs = np.array(idxs)            
        idx_to_class = {val:key for key, val in model.class_to_idx.items()}
        classes = [idx_to_class[idx] for idx in idxs[0]]
        
        # map the class name with collected topk classes
        names = []
        for cls in classes:
            names.append(cat_to_name[str(cls)])
        
        return np.exp(prob[0]), names

image_pil = Image.open(image_path)
prob, names = predict(image_pil, model_load, top_k)
print("prob: ", prob, "   namee: ", names)

print("done")


 