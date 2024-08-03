import os
import cxr_dataset as CXR
import eval_model as E
import model as M
import argparse
import torch

os.chdir("./")

parser = argparse.ArgumentParser('Train Transformer Model')
parser.add_argument('--images_path', default='/images/')
parser.add_argument('--mode', default='train')
args = parser.parse_args()

mode = args.mode
PATH_TO_IMAGES = args.images_path
WEIGHT_DECAY = 1e-4
LEARNING_RATE = 0.0001
EPOCHS = 50
BATCH_SIZE = 32

if mode == "train":
    model, best_epoch = M.train_transformer(PATH_TO_IMAGES, LEARNING_RATE, WEIGHT_DECAY, EPOCHS, BATCH_SIZE)
elif mode == "eval":
    # load model weights for eavluation
    checkpoint_best = torch.load('results/checkpoint')
    model = checkpoint_best['model']
    
preds, aucs = E.make_pred_multilabel(model, PATH_TO_IMAGES)