#Imports
import torch
from torchvision import datasets, transforms, models
from torch import optim, nn
import json
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
from matplotlib import interactive
interactive(True)
import numpy as np
import torch.nn.functional as F
import os
os.environ['QT_QPA_PLATFORM']='offscreen'
import argparse, sys


#reading in data sets
data_dir = 'flowers'
train_dir = data_dir + '/train'
valid_dir = data_dir + '/valid'
test_dir = data_dir + '/test'


#defining data transforms
data_transforms = {'train_set':transforms.Compose([transforms.RandomResizedCrop(224), transforms.RandomRotation(30), 
                                                  transforms.RandomHorizontalFlip(), transforms.ToTensor(),
                                                 transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
                   'valid_set' :transforms.Compose([transforms.Resize(255), transforms.CenterCrop(224), 
                                                  transforms.ToTensor(),
                                                  transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
                   'test_set' :transforms.Compose([transforms.Resize(255), transforms.CenterCrop(224),
                                                      transforms.ToTensor(), 
                                                     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
                  }

# Load the datasets with ImageFolder
data_dir = {
    'train_set': train_dir,
    'valid_set': valid_dir,
    'test_set': test_dir
}
image_datasets = {x: datasets.ImageFolder(data_dir[x], transform = data_transforms[x]) 
                  for x in ['train_set', 'valid_set', 'test_set'] }


