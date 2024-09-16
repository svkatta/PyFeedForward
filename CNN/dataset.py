import pandas as pd 
from torch.utils.data import Dataset
import torch 
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import torchvision

class TinyImageNet(Dataset):
    def __init__(self, path, shape , transform=None, target_transform=None):
        self.imgs , self.img_labels = self.read_data(path,shape)
        self.transform = transform
        self.target_transform = target_transform
        self.num_classes = 20

    def __len__(self):
        return len(self.imgs)

    def convert_to_onehot(self,indices, num_classes):
        output = np.eye(num_classes)[np.array(indices).reshape(-1)]
        result = output.reshape(list(np.shape(indices))+[num_classes])
        return result    

    def __getitem__(self, idx):
        image = self.imgs[idx]
        image = np.transpose(image , (2,0,1))
        label = self.img_labels[idx]
        return  torch.from_numpy(image), label

    def read_data(self,path, shape):
        h = shape[0]
        w = shape[1]
        c = shape[2]
        data = pd.read_csv(path)
        data = data.to_numpy()
        # TODO: (0) Check this normalization and look for a better one
        X = (data[:, 1:-1])/255
        # TODO: (0) Check this reshaping 
        X = X.reshape(-1,h,w,c)
        Y = data[:,-1]
        #Y = self.convert_to_onehot(Y, num_classes)
        print("Shape of data: ", np.shape(X), "Shape of train labels: ", np.shape(Y))
        # print("Shape of data: ", np.type(X), "Shape of train labels: ", np.type(Y))
        return   np.float32(X)  ,   Y     
