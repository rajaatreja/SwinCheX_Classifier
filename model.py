from __future__ import print_function, division

# pytorch imports
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
import torchvision
from torchvision import datasets, models, transforms
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

# image imports
from skimage import io, transform
from PIL import Image

# general imports
import os
import time
from shutil import copyfile
from shutil import rmtree
import copy
# data science imports
import pandas as pd
import numpy as np
import csv

import cxr_dataset as CXR
import eval_model as E
import torch.utils.checkpoint as cp
import torch.nn.functional as F
import timm

use_gpu = torch.cuda.is_available()
gpu_count = torch.cuda.device_count()
print("Available GPU count:" + str(gpu_count))


def checkpoint(model, best_loss, epoch, LR):
    """
    Saves checkpoint of torchvision model during training.

    Args:
        model: torchvision model to be saved
        best_loss: best val loss achieved so far in training
        epoch: current epoch of training
        LR: current learning rate in training
    Returns:
        None
    """

    print('saving')
    state = {
        'model': model,
        'best_loss': best_loss,
        'epoch': epoch,
        'rng_state': torch.get_rng_state(),
        'LR': LR
    }

    torch.save(state, 'results/checkpoint')


def train_model(model, criterion, optimizer, scheduler, num_epochs, dataloaders, dataset_sizes, patience=3):
    """
    Fine-tunes a model on NIH CXR data with advanced early stopping.

    Args:
        model: The model to be fine-tuned.
        criterion: The loss criterion to be used.
        optimizer: The optimizer for training.
        scheduler: The learning rate scheduler.
        num_epochs: The number of epochs to train for.
        dataloaders: Dataloaders for the training and validation datasets.
        dataset_sizes: Sizes of the training and validation datasets.
        patience: Number of epochs to wait after last time validation loss improved before stopping the training.

    Returns:
        model: The fine-tuned model.
        best_epoch: The epoch number with the best validation loss.
    """
    since = time.time()

    best_loss = float('inf')
    best_model_wts = copy.deepcopy(model.state_dict())
    epochs_since_improvement = 0
    best_epoch = 0

    for epoch in range(1, num_epochs + 1):
        print(f'Epoch {epoch}/{num_epochs}')
        print('-' * 10)

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0

            for inputs, labels, _ in dataloaders[phase]:
                inputs = inputs.to('cuda')
                labels = labels.to('cuda').float()

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)

            epoch_loss = running_loss / dataset_sizes[phase]
            print(f'{phase} Loss: {epoch_loss:.4f}')

            if phase == 'val':
                scheduler.step(epoch_loss)
                if epoch_loss < best_loss:
                    best_loss = epoch_loss
                    best_epoch = epoch
                    best_model_wts = copy.deepcopy(model.state_dict())
                    checkpoint(model, best_loss, epoch, optimizer.param_groups[0]['lr'])  # Save the best model
                    epochs_since_improvement = 0
                else:
                    epochs_since_improvement += 1

        print()

        if epochs_since_improvement == patience:
            print(f'Early stopping triggered after {epoch} epochs.')
            break

    time_elapsed = time.time() - since
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best val Loss: {best_loss:.4f}')

    model.load_state_dict(best_model_wts)
    return model, best_epoch


def train_transformer(PATH_TO_IMAGES, LR, WEIGHT_DECAY, NUM_EPOCHS, BATCH_SIZE):
    """
    Train torchvision model to NIH data given high level hyperparameters.

    Args:
        PATH_TO_IMAGES: path to NIH images
        LR: learning rate
        WEIGHT_DECAY: weight decay parameter for SGD

    Returns:
        preds: torchvision model predictions on test fold with ground truth for comparison
        aucs: AUCs for each train,test tuple

    """

    try:
        rmtree('results/')
    except BaseException:
        pass  # directory doesn't yet exist, no need to clear it
    os.makedirs("results/")

    # use imagenet mean,std for normalization
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    N_LABELS = 14  # we are predicting 14 labels

    # load labels
    df = pd.read_csv("nih_labels.csv", index_col=0)

    # define torchvision transforms
    data_transforms = {
    'train': transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]),
    # Validation transforms remain the same
    'val': transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
}

    # create train/val dataloaders
    transformed_datasets = {
        'train': CXR.CXRDataset(
            path_to_images=PATH_TO_IMAGES,
            fold='train',
            transform=data_transforms['train']),
        'val': CXR.CXRDataset(
            path_to_images=PATH_TO_IMAGES,
            fold='val',
            transform=data_transforms['val']),
    }

    dataloaders = {
        'train': torch.utils.data.DataLoader(
            transformed_datasets['train'], batch_size=BATCH_SIZE, shuffle=True, num_workers=8),
        'val': torch.utils.data.DataLoader(
            transformed_datasets['val'], batch_size=BATCH_SIZE, shuffle=True, num_workers=8),
    }


    # please do not attempt to train without GPU as will take excessively long
    if not use_gpu:
        raise ValueError("Error, requires GPU")

    # Initialize Swin Transformer
    model = timm.create_model('swin_large_patch4_window7_224', pretrained=True, num_classes=N_LABELS)  # 14 head

    # put model on GPU
    model = model.cuda()

    # define criterion, optimizer for training
    criterion = nn.BCEWithLogitsLoss()  # Update loss function for multilabel
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    # Introducing a ReduceLROnPlateau learning rate scheduler
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5)
    dataset_sizes = {x: len(transformed_datasets[x]) for x in ['train', 'val']}

    # train model
    model, best_epoch = train_model(model, criterion, optimizer, scheduler, NUM_EPOCHS, dataloaders, dataset_sizes)

    return model, best_epoch
