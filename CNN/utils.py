import argparse
import torch
import random
import numpy as np

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def get_arg_parser():
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
    
    return parser

def set_seed(seed = 31):
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def move_to_gpu(*args):
    if torch.cuda.is_available():
        for item in args:
            item.cuda()
