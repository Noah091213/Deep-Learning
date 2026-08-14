import torch
import torch.nn as nn

input_dim = (224, 224) # This scales the image so it can be an input for the network
channel_dim = 1 # Greyscale, xray doesnt give any color information anyway

class group_1(nn.Module):
    def __init__(self):
        super(group_1, self).__init__()
        # Define channels and kernel sizes for dynamic model construction thingy
        # This makes it easier to tune and lower total parameters
        conv_channels = [17, 13, 9, 5] # Every entry is one layer
        kernel_sizes = [3, 3, 3, 3] # Specify the kernel size for every layer

        layers = []
        prev_channels = channel_dim
        for out_channels, k in zip(conv_channels, kernel_sizes):
            # Dynamically building the layers of the CNN, every layer has these 4 functions
            layers.append(nn.Conv2d(prev_channels, out_channels, kernel_size=k))
            layers.append(nn.BatchNorm2d(out_channels))
            layers.append(nn.ReLU())
            layers.append(nn.MaxPool2d(kernel_size=3))
            prev_channels = out_channels
        self.conv_layers = nn.Sequential(*layers) # Package the convolution layers for use

        # This just checks what the size of the flattened layer is with a dummy tensor
        with torch.no_grad():
            dummy = torch.zeros(1, channel_dim, *input_dim)
            flat_size = self.conv_layers(dummy).numel()

        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(flat_size, 64)
        self.dropout = nn.Dropout(0.2)
        self.act = nn.ReLU()
        self.fc2 = nn.Linear(64, 2) # 2 neurons for picking normal or pneumonia

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.act(self.dropout(x))
        x = self.fc2(x) # Cross Entropy Loss brings the softmax
        return x
