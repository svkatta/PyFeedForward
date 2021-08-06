import argparse
import matplotlib.pyplot as plt 
import numpy as np 
import os
import pandas as pd
import pickle 
from sklearn.decomposition import PCA
import skimage
import sys
import torch 
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import torchvision


from model import CNN_Net 


parser = argparse.ArgumentParser()
parser.add_argument("--lr", default = 0.001, help = "learning rate, defaults to 0.01", type = float)
parser.add_argument("--batch_size", default = 256, help = "size of each minibatch, defaults to 256", type = int)
parser.add_argument("--init", default = 1, help = "initialization to be used; 1: Xavier; 2: He; defaults to 1", type = int)
parser.add_argument("--save_dir", help = "location for the storage of the final model")
parser.add_argument("--epochs", default = 10, help = "number of epochs", type = int)
parser.add_argument("--dataAugment", default = 0, help = "1: use data augmentation, 0: do not use data augmentation", type = int)
parser.add_argument("--train", default = "train.csv", help = "path to the training data")
parser.add_argument("--val", default = "valid.csv", help = "path to the validation data")
parser.add_argument("--test", default = "test.csv", help = "path to the test data")
args = parser.parse_args()



#####---------------DataSet-----------#####

class TinyImageNet(Dataset):
    def __init__(self, path, shape , transform=None, target_transform=None):
        self.imgs , self.img_labels = self.read_data(path,shape)
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.imgs)

    def convert_to_onehot(self,indices, num_classes):
        # borrowed from stack overflow
        output = np.eye(num_classes)[np.array(indices).reshape(-1)]
        # the reshape just seems to verify the shape of the matrix
        # each target vector is converted to a row vector
        result = output.reshape(list(np.shape(indices))+[num_classes])
        return result    

    def read_data(self,path, shape):
        h = shape[0]
        w = shape[1]
        c = shape[2]
        num_classes = 20
        data = pd.read_csv(path)
        data = data.to_numpy()
        # TODO: (0) Check this normalization and look for a better one
        X = (data[:, 1:-1])/255
        # TODO: (0) Check this reshaping 
        X = X.reshape(-1,h,w,c)
        Y = data[:,-1]
        #Y = self.convert_to_onehot(Y, num_classes)
        print("Shape of data: ", np.shape(X), "Shape of train labels: ", np.shape(Y))
        return  X ,  Y    

    def __getitem__(self, idx):
        image = self.imgs[idx]
        image = np.transpose(image , (2,0,1))
        label = self.img_labels[idx]
        return  torch.from_numpy(image), label

train_dataset = TinyImageNet(path=args.train,shape=(64,64,3))
valid_dataset = TinyImageNet(path=args.valid,shape=(64,64,3))       

train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
test_dataloader = DataLoader(valid_dataset, batch_size=4, shuffle=False)

#####---------------Model-----------#####
net = CNN_Net()

if torch.cuda.is_available:
    net = net.cuda()

#####---------------traning Loop-----------#####


criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9)
for epoch in range(args.batch_size):  # loop over the dataset multiple times
    running_loss = 0.0
    for i, data in enumerate(train_dataloader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data
        inputs, labels = inputs.cuda().to(dtype=torch.float) , labels.cuda().to(dtype=torch.long)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i % 2000 == 1999:    # print every 2000 mini-batches
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 2000))
            running_loss = 0.0


def accuracy(data_loader): 
    with torch.no_grad():
        correct=0
        total =0 
        for data in data_loader:
            inputs, labels = data
            if torch.cuda.is_available:
                inputs, labels = inputs.cuda().to(dtype=torch.float) , labels.cuda().to(dtype=torch.long)
            
            outputs = net(inputs)
            predicitons = torch.argmax(outputs,1)
            correct = correct + torch.sum(predicitons == labels)
            total = total + labels.size()[0]
    return correct/total