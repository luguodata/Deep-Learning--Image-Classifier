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


# Add data directory
parser = argparse.ArgumentParser(description = 'This is Impage Classifier Traning File')
parser.add_argument('data_dir', help = 'Train test data folders')
parser.add_argument('--save_dir', help = 'Check Point Save Directory')
parser.add_argument('--arch', help = 'pre_train model archtecture', choices = ['vgg16_bn','vgg16','vgg13'])
parser.add_argument('--learning_rate', help = 'training learning rate' , type = float)
parser.add_argument('--input_size', help = 'training network input size' , type = int)
parser.add_argument('--hidden_sizes', help = 'training network hidden layer sizes', nargs = '+', type = int )
parser.add_argument('--output_size', help = 'training network output size' , type = int)
parser.add_argument('--drop_p', help = 'training network drop out probab' , type = float)
parser.add_argument('--epochs', help = 'training epochs' , type = int)
parser.add_argument('--gpu', help = 'GPU/CPU', choices = ['cuda','cpu'] )


args = parser.parse_args()



# define data folders
train_dir = args.data_dir + '/train'
valid_dir = args.data_dir + '/valid'
test_dir = args.data_dir + '/test'


# TODO: Define your transforms for the training, validation, and testing sets
train_data_transforms = transforms.Compose([
                                      transforms.Resize(256),
                                      transforms.CenterCrop(224),
                                      #transforms.RandomRotation(30),
                                      #transforms.RandomHorizontalFlip(),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485,0.456,0.406],
                                                           [0.229,0.224,0.225])])

valid_test_data_transforms = transforms.Compose([
                                                 transforms.Resize(256),
                                                 transforms.CenterCrop(224),
                                                 transforms.ToTensor(),
                                                 transforms.Normalize([0.485,0.456,0.406],
                                                                      [0.229,0.224,0.225])])

# TODO: Load the datasets with ImageFolder
image_datasets_train = datasets.ImageFolder(train_dir, transform = train_data_transforms)
image_datasets_validation = datasets.ImageFolder(valid_dir, transform = valid_test_data_transforms)
image_datasets_test = datasets.ImageFolder(test_dir, transform = valid_test_data_transforms)



# TODO: Using the image datasets and the trainforms, define the dataloaders
train_dataloaders = torch.utils.data.DataLoader(image_datasets_train, batch_size= 64, shuffle= True)
valid_dataloaders = torch.utils.data.DataLoader(image_datasets_validation, batch_size= 32, shuffle= True)
test_dataloaders = torch.utils.data.DataLoader(image_datasets_test, batch_size= 32, shuffle= True)


# impot category to name mapping
with open('cat_to_name.json', 'r') as f:
    cat_to_name = json.load(f)


# set new feed-ward classifier

# select pre-train model, 3 availables
if args.arch == 'vgg16_bn':
    pre_model = models.vgg16_bn(pretrained= True)
elif args.arch == 'vgg13':
    pre_model = models.vgg13(pretrained= True)
elif args.arch == 'vgg16':
    pre_model = models.vgg16(pretrained= True)


flwer_model = network_build(pre_model, 25088, args.hidden_sizes,  121, 0.1)
flwer_model = image_training_validation(args.epochs,flwer_model,train_dataloaders,valid_dataloaders,nn.NLLLoss(),args.learning_rate,args.gpu)


# TODO: Do validation on the test set
with torch.no_grad():
    test_loss, test_accu = validation_test(flwer_model, test_dataloaders, nn.NLLLoss(),args.gpu)

print('test dataset accuracy is : {}'.fotmat(test_accu))


# save clss_to_idx into model
flwer_model.class_to_idx = image_datasets_train.class_to_idx


# TODO: Write a function that loads a checkpoint and rebuilds the model
# TODO: Save the checkpoint
# Save Model
n_line = len(flwer_model.classifier)
hiddensizes = []

for n in range(n_line):
    # check if is the 'Linear' object
    if n % 3 == 0:
        # check if the first
        if n == 0:
            # get the input size
            inputsize = flwer_model.classifier[n].in_features
        # check if the last for getting output size
        elif n == n_line - 2:
            # get the last hidden layer's size
            hiddensizes.append(flwer_model.classifier[n].in_features)
            # get the output size
            outputsize = flwer_model.classifier[n].out_features

        # get the rest hidden sizes info
        else:
            hiddensizes.append(flwer_model.classifier[n].in_features)

    # get the drop out probability
    if n == 2:
        dropprop = flwer_model.classifier[n].p


# Save Model
flwer_checkpoint = {'pre_trained_model':pre_model,
                    'input_size': inputsize,
                    'hidden_sizes': hiddensizes,
                    'output_size': outputsize,
                    'drop_prop': dropprop,
                    'state_dict': flwer_model.state_dict(),
                    'class_to_idx':flwer_model.class_to_idx,
                    'epochs': args.epochs}


torch.save(flwer_checkpoint,args.save_dir)
