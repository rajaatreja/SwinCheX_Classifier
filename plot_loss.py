import matplotlib.pyplot as plt
import re

# Initialize lists to hold the extracted values
epochs = []
train_losses = []
val_losses = []

# Path to your file
file_path = '/home/hpc/iwi5/iwi5156h/swin/results/log_train'

# Regular expression to extract numbers from the tensor strings
pattern = re.compile(r'tensor\((\d+\.\d+), device=\'cuda:0\'\)')

# Read the file
with open(file_path, 'r') as file:
    for line in file:
        # Use the regex pattern to find all numeric values in the line
        matches = pattern.findall(line)
        
        if matches:
            # Assuming the epoch number is always correctly positioned before the tensor values
            # Split by comma and take the first part as epoch, which is then stripped of quotes and whitespace
            epoch = line.split(',')[0].strip()
            
            # Append the values to the respective lists
            epochs.append(int(epoch))
            train_losses.append(float(matches[0]))  # First match is train_loss
            val_losses.append(float(matches[1]))    # Second match is val_loss

# Plotting
plt.figure(figsize=(10, 6))
plt.plot(epochs, train_losses, label='Training Loss')
plt.plot(epochs, val_losses, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.grid(True)

# Save the plot to a file
plt.savefig('/home/hpc/iwi5/iwi5156h/swin/results/training_validation_loss_plot.png')
