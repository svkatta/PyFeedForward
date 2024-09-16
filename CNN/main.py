import matplotlib.pyplot as plt 
import numpy as np 
import os
import pandas as pd
 
import sys
import torch 
import torch.nn as nn

import torch.optim as optim
from torch.utils.data import DataLoader



from dataset import TinyImageNet
from utils import move_to_gpu , get_arg_parser , set_seed
from model import CNN_Net 

def train(args):
    train_dataset = TinyImageNet(path=args.train,shape=(64,64,3))
    valid_dataset = TinyImageNet(path=args.valid,shape=(64,64,3))       

    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_dataloader = DataLoader(valid_dataset, batch_size=4, shuffle=False)


    net = CNN_Net()

    move_to_gpu(net)
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


def main():
    set_seed()
    parser = get_arg_parser()
    args = parser.parse_args()
    print(args)
    train(args)

if __name__== "__main__":
    main()