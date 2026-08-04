from torchvision import datasets, transforms
from torchvision.transforms.functional import to_pil_image

transform = transforms.ToTensor()
test_dataset = datasets.MNIST(root='./data', train=False,
                               download=True, transform=transform)

# Grab the first N images and save them as PNGs, named with their true label
N = 10
for i in range(N):
    image_tensor, label = test_dataset[i]
    pil_image = to_pil_image(image_tensor)  # converts tensor back to a normal image
    pil_image.save(f'/home/ckhrix/Documents/DLVSC/Lecture 2/data/MNIST Samples/mnist_sample_{i}_label{label}.png')
    print(f"Saved mnist_sample_{i}_label{label}.png")