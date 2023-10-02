# PyFeedforward

Python Implementation of Feed Forward Neural Networks using Numpy

# Usage

`python train.py  <options>`

## Options
**Architecture** 
* `--num_hidden` : number of hidden layers
* `--sizes`      : comma separated list for the size of each hidden layer
* `--activation` : the choice of activation function - valid values are :`sigmoid`, `cross_entropy_softmax`, `tanh`, `softmax`
* `--PCA`        :  no of pca componets to be selected, PCA is performed on input data before feeding to Neural Network 

**Training**
* `--lr`         : initial learning rate for gradient descent based algorithms
* `--momentum`   : momentum to be used by momentum based algorithms
* `--loss`       : loss function to use while traning, supports : `cross_entropy[ce]`, `squared error[sq]`
* `--opt`        : the optimization algorithm to be used: `gd`, `momentum`, `nag`, `adam`
* `--batch_size` : the batch size to be used - valid values are 1 and multiples of 5
* `--epochs`     : number of passes over the data
* `--anneal`     : if true the algorithm should halve the learning rate if at any epoch the validation error increases and then restart that epoch
* `--pretrain`   : Flag to use pre train model

**Data and Model**
* `--save_dir`: the directory in which the pickled model should be saved
* `--expt_dir`: the directory in which the log files will be saved
* `--train`   : path to the training dataset
* `--val`     : path to the validation dataset
* `--test`    : path to the test dataset

**Others**
* `--testing` : Flag to test model
* `--logs`    : Flag to either write logs into a file or not 


Example Usage : `python train.py --lr 0.01 --momentum 0.9 --num_hidden 2 --sizes 240,240 --activation sigmoid --loss ce --opt adam --batch_size 20 --epochs 20 --anneal True --save_dir ../save_dir/best/ --expt_dir ../expt_dir/ --train train.csv --val valid.csv --test test.csv --pretrain False --state 20 --testing False--logs False --PCA 40`

