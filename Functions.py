
import torch
from torchvision import datasets, transforms
import torchvision.models as models
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from collections import OrderedDict
import argparse
import json
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt


def network_build(pre_train_model, input_size, hidden_sizes, output_size, drop_p):
    """ Load pre-trained network and construct new untrained feed-forward network as the classifer.
    INPUT: pre_train_model, the selected pre-trained network from TorchVision.
           input_size, int, the input size of the untrained feed-forward classifier.
           hidden_sizes: list of int to represnet sizes of the network's hidden layer.
           output_size, int,  the output size of the untrained feed-forward classifier

    OUTPUT: pre-trained model with new feed-forward classifer.
    """

    #initialize pre-tarined model
    model = pre_train_model

    ## to be filled
    for para in model.parameters():
        para.requires_grad = False

    # intialize an empty newwork layer list for contraining new classifier
    network_layers = []

    # first layer
    network_layers.append(('fc1', nn.Linear(input_size, hidden_sizes[0])))
    network_layers.append(('ReLu1', nn.ReLU()))
    network_layers.append(('Dropout1', nn.Dropout(drop_p)))

    # hidden layers
    for n in range(len(hidden_sizes)):

        if n == (len(hidden_sizes) - 1):

            network_layers.append(('fc_out', nn.Linear(hidden_sizes[n], output_size)))
            network_layers.append(('out_log', nn.LogSoftmax(dim = 1)))

        else:
            network_layers.append(('fc' + str(n+2), nn.Linear(hidden_sizes[n], hidden_sizes[n+1])))
            network_layers.append(('ReLu' + str(n+2), nn.ReLU()))
            network_layers.append(('Dropout' + str(n+2) , nn.Dropout(drop_p)))

    # construct feed-forward network with layer list
    img_network = nn.Sequential(OrderedDict(network_layers))
    # set as pre-trained network's classifer
    model.classifier = img_network

    return model


def validation_test(trained_model, dataloaders_check, criterion, device):
    """ Use trained image model to do prediction on validation dataset
    INPUT: trained_model, the pretrained model with new classifier layers
           dataloaders_check, validation/test dataloaders
           criterion: used to calculate loss
           device: 'cpu'/'cuda'
    OUTPUT: validation/testing loss & accuracy
    """
    # set as evaluation mode
    trained_model.eval()

    # initialize metrics
    val_test_loss = 0
    total_img = 0
    val_test_correct_num = 0

    for val_test_images, val_test_labels in iter(dataloaders_check):

        val_test_images, val_test_labels = val_test_images.to(device), val_test_labels.to(device)

        # forward output
        val_test_output = trained_model.forward(val_test_images)
        # transform logged result to original
        val_test_exp = torch.exp(val_test_output)
        # record val_testidation set los
        val_test_loss += criterion(val_test_output, val_test_labels).item()

        # record pred correction number
        total_img += len(val_test_images)
        val_test_correct_num += (val_test_exp.max(dim = 1)[1] == val_test_labels.data).type(torch.FloatTensor).sum()

    val_test_accuracy = val_test_correct_num/total_img

    return val_test_loss, val_test_accuracy


def image_training_validation(epochs,model,dataloader_train, dataloader_valid, criterion, img_lr, device):
    """ Train image NN model and do validation during training process
    INPUT: epochs, int, number of training rounds
           model, pre-trained model
           dataloader_train, training dataset
           dataloader_valid, validation dataset
           criterion: used for calculating loss
           img_lr: learning rate
           device: 'cpu'/'cuda'
    OUTPUT: trained model, print out training loss, validation loss and validation accuracy
    """
    # set optimizer
    img_optimizer = optim.Adam(model.classifier.parameters(), lr = img_lr)

    # defining initial metrics
    training_loss = 0
    training_loss = 0
    print_fold = 40

    steps = 0

    model.to(device)

    for e in range(epochs):

        for images, labels in iter(dataloader_train):

            # set model as traning mode
            model.train()

            images, labels = images.to(device), labels.to(device)

            steps += 1

            img_optimizer.zero_grad()
            # make forward feed
            output = model.forward(images)

            # take loss
            loss = criterion(output, labels)

            # backwards
            loss.backward()

            # aggregate training loss
            training_loss += loss.item()

            # update weights
            img_optimizer.step()


            # print training loss and test on validataion at every print_fold
            if steps % print_fold == 0:

                # output training loss
                print('In No. {} epoch'.format(e+1))
                print('training loss: {:.4f}'.format(training_loss))

                # test on validation set
                # set model as validation mode
                with torch.no_grad():
                    val_loss, val_accur = validation_test(model, dataloader_valid, criterion, device)

                # print validation loss
                print('validation loss: {:.4f}'.format(val_loss))

                # print validation accuracy
                print('validation accuracy: {:.2f}'.format(val_accur))

                # reset training loss
                training_loss = 0

                # rest to train mode
                model.train()

    return model



# TODO: Write a function that loads a checkpoint and rebuilds the model
def load_flwer_model(flwer_file_path):
    check_point = torch.load(flwer_file_path)

    model = network_build(check_point['pre_trained_model'], check_point['input_size'], check_point['hidden_sizes']\
                          ,check_point['output_size'], check_point['drop_prop'])

    model.load_state_dict(check_point['state_dict'])
    model.class_to_idx = check_point['class_to_idx']

    return model



def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''

    # TODO: Process a PIL image for use in a PyTorch model

    # resize
    w, h = image.size
    if w < h:
        w_r = 256
        h_r = int(w_r/w * h)
    else:
        h_r = 256
        w_r = int(h_r/h * w)

    image_resize = image.resize((w_r,h_r))


    # center crop
    left_w = (w_r - 224)/2
    right_w = (w_r + 224)/2
    top_h = (h_r - 224 )/2
    botm_h = (h_r + 224)/2

    image_crop = image_resize.crop((left_w,top_h,right_w,botm_h))

    # convert to 0-1
    np_image = np.array(image_crop)/ 255

    normal_image = (np_image - np.array([0.485, 0.456, 0.406]))/np.array([0.229, 0.224, 0.225])

    # re-order
    reorder_img = normal_image.transpose((2,0,1))

    return reorder_img


def imshow(image, ax=None, title=None):
    if ax is None:
        fig, ax = plt.subplots()

    # PyTorch tensors assume the color channel is the first dimension
    # but matplotlib assumes is the third dimension
    image = image.transpose((1, 2, 0))

    # Undo preprocessing
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = std * image + mean

    # Image needs to be clipped between 0 and 1 or it looks like noise when displayed
    image = np.clip(image, 0, 1)

    ax.imshow(image)

    return ax


def predict(image_path, model, topk, cat_name , device):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''

    # TODO: Implement the code to predict the class from an image file
    model.eval()
    model.to(device)
    model.double()

    img = process_image(Image.open(image_path))

    with torch.no_grad():
        # forward output
        pred_ori = model.forward(torch.from_numpy(img).unsqueeze_(0))
        # transform logged result to original
        pred = torch.exp(pred_ori)

        prob, idx = pred.topk(topk)

    #
    spc_idx_class = []
    spc_idx_names = []

    for key, value_1 in model.class_to_idx.items():
        for value_2 in  idx.numpy()[0].tolist():
            if value_1 == value_2:
                spc_idx_class.append(key)

    for cat in spc_idx_class:
        spc_idx_names.append(cat_name[cat])

    return spc_idx_names, prob.numpy()[0].tolist()
