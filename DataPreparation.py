import pandas as pd
import numpy as np
import os
# Torch and Torchvision
import torchvision 
#from torch.utils import data
from torch.utils.data import Dataset, TensorDataset,DataLoader
from torchvision import datasets
import torchvision.transforms as transforms 
import matplotlib.pyplot as plt
from PIL import Image
import torch

## Data Preparation

### Load Data and check Meta info

transforms_raw = transforms.Compose([transforms.ToTensor()]) #for conversion to tensors
trainset = torchvision.datasets.CIFAR10(
    root='./data', train=True, download=True, transform=transforms_raw)
testset = torchvision.datasets.CIFAR10(
    root='./data', train=False, download=True, transform=transforms_raw)

# get train, test set sizes
trainset_size =len(trainset) 
testset_size = len(testset)
print(trainset_size, testset_size)

#trainset.targets

# meta data information
trainset.class_to_idx

# Calculate mean and std of the pixel values in images and nomalize it while transforming
x = np.concatenate([np.asarray(
    trainset[i][0].reshape(-1,
                             trainset[0][0].shape[0],
                             trainset[0][0].shape[1],
                             trainset[0][0].shape[2])
    ) for i in range(len(trainset))])
# print(x)
print(x.shape)

#trainset.data[0]
#trainset[0][0]

# calculate the mean and std along the (0, 1) axes
train_mean = np.mean(x, axis=(0,2,3))
train_std = np.std(x, axis=(0,2,3))
print(train_mean)
print(train_std)

### Filter 50% of the labels for training (bird, deer and truck)

# traning data labels for bird = 2, deer =4, truck = 9 need to be cut down to half 
custom_data=[]
custom_label =[]
bird = 0
deer =0
truck = 0
limit = 2500
for i in range(len(trainset)):#range(20):
    train_data_i = trainset[i][0]
    train_label_i = trainset[i][1]
    rules = [train_label_i == 0,
             train_label_i == 1,
             train_label_i == 3,
             train_label_i == 5,
             train_label_i == 6,
             train_label_i == 7,
             train_label_i == 8]
    if any(rules):
        custom_data.append(train_data_i)
        custom_label.append(train_label_i)      
    elif train_label_i==2:
      if bird<limit:
        bird+=1
        custom_data.append(train_data_i)
        custom_label.append(train_label_i)
    elif train_label_i==4:
      if deer<limit:
        deer+=1
        custom_data.append(train_data_i)
        custom_label.append(train_label_i)
    elif train_label_i==9:
      if truck<limit:
        truck+=1
        custom_data.append(train_data_i)
        custom_label.append(train_label_i)        


print(len(custom_data))
#tensor_clabel = torch.Tensor(custom_label)
#tensor_cdata = torch.Tensor(custom_data)

print(len(custom_data))
tensor_clabel = custom_label
tensor_cdata = custom_data



class CustomTensorDataset():
    """TensorDataset with support of transforms.
    """
    def __init__(self, images,labels, transform=None):
        self.images = images
        self.labels = labels
        self.transform = transform

    def __getitem__(self, index):
        labels = self.labels[index]
        image_tensor = self.images[index]
        images = transforms.ToPILImage()(image_tensor)

        if self.transform:
            images = self.transform(images)


        return images, labels

    def __len__(self):
        return len(self.labels)

train_transform = transforms.Compose([transforms.RandomHorizontalFlip(),
                                      transforms.RandomRotation(10),
                                      transforms.ToTensor(),
                                      transforms.Normalize(train_mean, train_std)
                                     ])

test_transform = transforms.Compose([transforms.ToTensor(),
                                     transforms.Normalize(train_mean, train_std)
                                    ])

# we will use a rough 80% train, 10% val, 10% test spilt 
cifar10datasets = {
    'train':CustomTensorDataset(images=tensor_cdata,labels=tensor_clabel,
     transform=train_transform)
            }
valtest_dataset= datasets.CIFAR10(
         root='./data', train=False, download=True, 
         transform=test_transform)
cifar10datasets['test'], cifar10datasets['val'] = torch.utils.data.random_split(valtest_dataset, [5000, 5000])


### Build  Data Iterators


batch_size=16
dataloaders = {
'train' : DataLoader(cifar10datasets['train'], batch_size=batch_size,shuffle=True, num_workers=2),
'test' : DataLoader(cifar10datasets['test'], batch_size=batch_size,shuffle=True, num_workers=2),
'val' : DataLoader(cifar10datasets['val'], batch_size=batch_size,shuffle=True, num_workers=2)}



