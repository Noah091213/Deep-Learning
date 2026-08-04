import torch
import torch.nn as nn
from torchvision import datasets, transforms
from PIL import Image

# --- Same architecture as training script ---
# Must match exactly, since load_state_dict just pours saved numbers
# into layers of these exact shapes.
class MyNetwork(nn.Module):
    def __init__(self):
        super(MyNetwork, self).__init__()
        self.fc1 = nn.Linear(in_features=784, out_features=20)
        self.fc2 = nn.Linear(in_features=20, out_features=20)
        self.fc3 = nn.Linear(in_features=20, out_features=10)
        self.flatten = nn.Flatten()
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.flatten(x)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)  # raw logits, no softmax (matches training script)
        return x


# --- Load the trained model ---
model = MyNetwork()
model.load_state_dict(torch.load('model.pth'))
model.eval()  # good practice for inference, even though this network
              # has no train/eval-sensitive layers like dropout


# --- Choose your image source ---
# Set USE_MNIST_TEST = True to grab an image straight from the MNIST
# test set by index. Set it to False to load your own 28x28 image file.
USE_MNIST_TEST = False
MNIST_INDEX = 0                 # only used if USE_MNIST_TEST is True
CUSTOM_IMAGE_PATH = '/home/ckhrix/Documents/DLVSC/Lecture 2/data/MNIST Samples/mnist_sample_9_label9.png'  # only used if USE_MNIST_TEST is False

transform = transforms.ToTensor()

if USE_MNIST_TEST:
    test_dataset = datasets.MNIST(root='./data', train=False,
                                   download=True, transform=transform)
    image, true_label = test_dataset[MNIST_INDEX]
    print(f"True label: {true_label}")
else:
    # Convert to grayscale ('L') to match MNIST's single color channel
    pil_image = Image.open(CUSTOM_IMAGE_PATH).convert('L')
    image = transform(pil_image)  # scales pixel values to 0-1, shape [1, 28, 28]

# The model expects a batch dimension: [batch_size, 1, 28, 28].
# A single loaded image is [1, 28, 28], so add a dimension at position 0.
image = image.unsqueeze(0)


# --- Predict ---
with torch.no_grad():
    outputs = model(image)                       # raw logits, shape [1, 10]
    probabilities = torch.softmax(outputs, dim=1)  # convert to 0-1 confidences
    _, predicted = torch.max(outputs, dim=1)

predicted_digit = predicted.item()
confidence = probabilities[0, predicted_digit].item()

print(f"Predicted digit: {predicted_digit}")
print(f"Confidence: {confidence * 100:.2f}%")