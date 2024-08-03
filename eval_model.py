import torch
from torch.autograd import Variable
import pandas as pd
import numpy as np
import cxr_dataset as CXR  # Assuming cxr_dataset is your custom module for dataset loading
from torchvision import transforms
import sklearn.metrics as sklm

def make_pred_multilabel(model, PATH_TO_IMAGES):
    """
    Gives predictions for test fold and calculates AUCs using previously trained model

    Args:
        model: Model to use for predictions.
        PATH_TO_IMAGES: path at which NIH images can be found.

    Returns:
        pred_df: dataframe containing individual predictions and ground truth for each test image.
        auc_df: dataframe containing aggregate AUCs by train/test tuples.
    """
    BATCH_SIZE = 32
    model.eval()  # Set model to evaluation mode

    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    data_transforms = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

    dataset = CXR.CXRDataset(
        path_to_images=PATH_TO_IMAGES,
        fold="test",
        transform=data_transforms)
    dataloader = torch.utils.data.DataLoader(dataset, BATCH_SIZE, shuffle=False, num_workers=8)

    pred_df = pd.DataFrame(columns=["Image Index"])
    true_df = pd.DataFrame(columns=["Image Index"])

    for i, data in enumerate(dataloader):
        inputs, labels, _ = data
        inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())

        with torch.no_grad():  # Inference without gradient calculation
            outputs = model(inputs)
            # Apply sigmoid to outputs to convert logits to probabilities
            probs = torch.sigmoid(outputs).cpu().data.numpy()

        true_labels = labels.cpu().data.numpy()
        batch_size = true_labels.shape[0]

        for j in range(batch_size):
            thisrow = {"Image Index": dataset.df.index[BATCH_SIZE * i + j]}
            truerow = {"Image Index": dataset.df.index[BATCH_SIZE * i + j]}
            
            for k, label in enumerate(dataset.PRED_LABEL):
                thisrow["prob_" + label] = probs[j, k]
                truerow[label] = true_labels[j, k]

            pred_df = pred_df.append(thisrow, ignore_index=True)
            true_df = true_df.append(truerow, ignore_index=True)

        if i % 10 == 0:
            print(f"{i * BATCH_SIZE} images processed.")

    auc_df = pd.DataFrame(columns=["label", "auc"])
    for column in true_df.columns[1:]:  # Skip 'Image Index'
        actual = true_df[column].values.astype(int)
        pred = pred_df["prob_" + column].values
        try:
            auc = sklm.roc_auc_score(actual, pred)
            auc_df = auc_df.append({"label": column, "auc": auc}, ignore_index=True)
        except ValueError as e:
            print(f"Can't calculate AUC for {column}: {e}")
    # Calculate the average AUC and append it to the auc_df
    average_auc = auc_df['auc'].mean()
    auc_df = pd.concat([auc_df, pd.DataFrame({'label': 'average_auc', 'auc': average_auc}, index=[0])], ignore_index=True)
    pred_df.to_csv("results/preds.csv", index=False)
    auc_df.to_csv("results/aucs.csv", index=False)

    return pred_df, auc_df
