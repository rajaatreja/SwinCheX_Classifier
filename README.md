
# SwinCheX Classifier

A deep learning model for classifying chest X-ray images from the ChestX-ray14 dataset using a modified Swin Transformer architecture.

## Overview

The **SwinCheX Classifier** is designed to classify 14 different diseases found in the ChestX-ray14 dataset. This project uses the [Swin Transformer](https://github.com/microsoft/Swin-Transformer) as the base model, which has been adapted for multi-class classification tasks specific to chest X-ray images.

### Key Features

- **Swin Transformer Backbone**: Utilizes the powerful Swin Transformer architecture for effective feature extraction.
- **14-Class Output**: Modified for the classification of 14 distinct chest diseases.
- **Pre-trained Weights**: Access to pre-trained model weights for faster deployment.
- **Customizable Training**: Flexible configuration options for training parameters.

## Table of Contents

- [Requirements](#requirements)
- [Dataset](#dataset)
- [Installation](#installation)
- [Usage](#usage)
  - [Training](#training)
  - [Inference](#inference)
- [Model Architecture](#model-architecture)
- [Results](#results)
- [License](#license)
- [Acknowledgments](#acknowledgments)

## Requirements

- **Python**: Ensure that Python is installed (version 3.7 or higher is recommended).
- **Packages**: Required packages are listed in `requirements.txt`. You can install them using:

  ```bash
  pip install -r requirements.txt
  ```

## Dataset

The model is trained and tested on the **ChestX-ray14** dataset, which contains images of various chest diseases. You can download the dataset from the [official NIH website](https://nihcc.app.box.com/v/ChestXray-NIHCC).

## Installation

1. **Clone the repository**:

   ```bash
   git clone https://github.com/your-username/swinchex-classifier.git
   cd swinchex-classifier
   ```

2. **Install dependencies**:

   ```bash
   pip install -r requirements.txt
   ```

3. **Download Trained Weights** (optional):

   If you want to use the already trained weights (named as pretrained_SwinCheX_classifier), download them from [here](https://github.com/rajaatreja/Pre-trained-Weights) and place them in the appropriate directory.

## Usage

### Training

Before training, configure the experiment parameters in `train_eval.py`. Key parameters include:

- **`WEIGHT_DECAY`**: Weight decay during training (default: `1e-4`).
- **`LEARNING_RATE`**: Learning rate for the optimizer (default: `0.0001`).
- **`EPOCHS`**: Number of training epochs (default: `50`).
- **`BATCH_SIZE`**: Batch size for training (default: `32`).

To train the model, run the following command:

```bash
python3 train_eval.py --mode "train" --images_path "/path_to_images/"
```

### Evaluation

To use the model for evaluation, execute the `train_eval.py` script:

```bash
python3 train_eval.py --mode "eval" --image_path "/path_to_image/"
```

The script will output the predicted class labels and confidence scores for each image.

## Model Architecture

The SwinCheX Classifier leverages the Swin Transformer architecture, known for its hierarchical representation and shifted windowing mechanism. The architecture has been adapted to fit the 14-class problem specific to the ChestX-ray14 dataset.

- **Hierarchical Design**: Efficiently models long-range dependencies with a pyramid structure.
- **Shifted Windowing**: Reduces computation by limiting self-attention to non-overlapping local windows.

## Results

The following table shows the AUC performance of the SwinCheX Classifier on the ChestX-ray14 dataset:

| Disease               | AUC           |
|-----------------------|---------------|
| Atelectasis           | 0.826         |
| Cardiomegaly          | 0.910         |
| Consolidation         | 0.815         |
| Edema                 | 0.891         |
| Effusion              | 0.883         |
| Emphysema             | 0.916         |
| Fibrosis              | 0.836         |
| Hernia                | 0.942         |
| Infiltration          | 0.718         |
| Mass                  | 0.858         |
| Nodule                | 0.787         |
| Pleural Thickening    | 0.780         |
| Pneumonia             | 0.763         |
| Pneumothorax          | 0.877         |

**Average AUC**: 0.843

These results highlight the model's strong performance across various diseases, demonstrating its effectiveness in the classification task.

## License

This project is licensed under the GNU General Public License (GPL). See the [LICENSE](LICENSE) file for more details.

## Acknowledgments

- The [Swin Transformer](https://github.com/microsoft/Swin-Transformer) team for their innovative architecture.
- The National Institutes of Health (NIH) for providing the ChestX-ray14 dataset.
- Community contributors and developers who have supported this project.