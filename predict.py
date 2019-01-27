# Imports here
import torch
from torchvision import datasets, transforms
import torchvision.models as models
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from collections import OrderedDict
import argparse
import json
from Functions import *
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt



# Add data directory
parser = argparse.ArgumentParser(description = 'This is Impage Classifier Traning File')
parser.add_argument('image_path', help = 'Check Point Save Directory')
parser.add_argument('check_point_path', help = 'check_point_path')
parser.add_argument('--top_k', help = 'top k predicted class', type = int)
parser.add_argument('--category_names', help = 'category_names')
parser.add_argument('--gpu', help = 'GPU/CPU', choices = ['cuda','cpu'] )


args = parser.parse_args()


with open(args.category_names, 'r') as f:
    cat_to_name = json.load(f)

    
# TODO: Write a function that loads a checkpoint and rebuilds the model
flwer_load_model = load_flwer_model(args.check_point_path)

# do prediction return name and probability
top_name, top_prob = predict(args.image_path, flwer_load_model, args.top_k, cat_to_name,args.gpu)

if args.top_k == 1:
    print('The predicted flower class is: {}, predicted probabilities is {}'.format(top_name, top_prob))
else:
    print('Top {} predicted flower classes are: {}, predicted probabilites are {}'.format(args.top_k, top_name, top_prob))
