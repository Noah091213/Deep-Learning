import torch
import torch.nn as nn

input_dim = (32, 32) # This scales the image so it can be an input for the network
channel_dim = 1 # Greyscale, xray doesnt give any color information anyway

class group_1(nn.Module):
    def __init__(self):
        super(group_1, self).__init__()
        # Define channels and kernel sizes for dynamic conv layer construction
        # This makes it easier to tune hyper parameters and lower total parameters
        conv_channels = [2, 7] # Every entry is one layer and the number specifies the amount of channels in that layer
        kernel_sizes = [3, 3, 3] # Specify the kernel size for each layer

        # This loop is for dynamically constructing the conv layers using the defined channels and kernels
        # Each loop builds one "conv layer" which includes everything in the loop
        layers = []
        prev_channels = channel_dim
        for i, (out_channels, k) in enumerate(zip(conv_channels, kernel_sizes)):
            if i == 0:
                # First layer input only has 1 channel, so depthwise separable conv doesnt do much
                layers.append(nn.Conv2d(prev_channels, out_channels, kernel_size=k, padding=1))
            else:
                # Depthwise separable conv followed by a pointwise conv (shoutout to MobileNets, check them out, very cool)
                layers.append(nn.Conv2d(prev_channels, prev_channels, kernel_size=k, padding=1, groups=prev_channels))
                layers.append(nn.Conv2d(prev_channels, out_channels, kernel_size=1))
            layers.append(nn.BatchNorm2d(out_channels))
            layers.append(nn.LeakyReLU())
            layers.append(nn.MaxPool2d(kernel_size=2))
            prev_channels = out_channels
        self.conv_layers = nn.Sequential(*layers) # Package the convolution layers for use

        self.global_pool = nn.AdaptiveAvgPool2d(1) # Avg pool instead of flattening to reduce params
        self.classifier = nn.Linear(prev_channels, 2)

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.global_pool(x)
        x = x.view(x.size(0), -1) # Drop the leftover 1x1 spatial dims, keep (batch, channels) for fc1
        x = self.classifier(x) # Cross Entropy Loss brings the softmax
        return x
    