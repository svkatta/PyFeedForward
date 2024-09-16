# CNNAssginment 4

This is a programming assignment in which convolutional neural networks are implemented and are trained in Pytorch. 

* Problem statement
* Train data
* Validation data
* Test data


## Contents
train.py

This script contains code for implementation and training of different convolutional neural networks for classification. The network to be trained must be changed within the code. Adam optimizer is used to train the network with cross entropy as the loss function. Data augmentation uses simple tricks such as flipping the images vertically, horizontally and rotating hte images.

Usage
Run as
```
python train.py --lr <learning_rate>  --batch_size <batch_size> --init <init> --save_dir <path_save_dir> --epochs <num_epochs> --dataAugment <augmentation> --train <path_to_train> --val <path_to_val> --test <path_to_test>
```

* learning_rate: learning rate to be used for all updates, defaults to 0.001
* batch_size: size of minibatch, defaults to 256
* init: initialization, 1 corresponds to Xavier and 2 corresponds to He initialization, defaults to 1
* path_save_dir: path to the folder where the final model is stored
* num_epochs: number of epochs to run for, defaults to 10
* augmentation: set to 0 for no augmentation, 1 for augmentation
* path_to_train: path to the training data .csv file
* path_to_val: path to the validation dataset .csv file
* path_to_test: path to the test dataset .csv file

Outputs


Note: ./ indicates that the file is created in the working directory

run.sh
A shell file containing the best set of hyperparameters for the given task. Run as described below to train a network with the specified architecture and predict values for the test data.

./run.sh