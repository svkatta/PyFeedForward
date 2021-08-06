import torch 
import torch.nn as nn

class CNN_Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, (5,5))
        self.conv2 = nn.Conv2d(32, 32, (5,5))
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv3 = nn.Conv2d(32, 64, (3,3))
        self.conv4 = nn.Conv2d(64, 64, (3,3))
        self.pool2 = nn.MaxPool2d(2, 2)
        self.conv5 = nn.Conv2d(64, 64, (3,3))
        self.conv6 = nn.Conv2d(64, 128, (3,3))
        self.pool3 = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(128*4*4, 256)
        # self.fc1 = nn.Linear(6272, 256)
        self.fc2 = nn.Linear(256, 20)

    def forward(self,x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x= self.pool1(x)
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x= self.pool2(x)
        x = F.relu(self.conv5(x))
        x = F.relu(self.conv6(x))
        x= self.pool3(x)

        x = torch.flatten(x, 1) 
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return x
