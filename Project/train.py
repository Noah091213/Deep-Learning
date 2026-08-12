import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os

from group_1 import group_1, input_dim, channel_dim


class CustomImageDataset(Dataset):
    def __init__(self, root_dir, channel_dim, transform=None):
        self.channel_dim = channel_dim
        self.transform = transform
        self.classes = sorted(os.listdir(root_dir))  # ['normal', 'pneumonia']
        self.label_to_idx = {label: idx for idx, label in enumerate(self.classes)}

        self.samples = []
        for label in self.classes:
            class_dir = os.path.join(root_dir, label)
            for f in os.listdir(class_dir):
                if f.lower().endswith(('.jpg', '.jpeg')):
                    self.samples.append((os.path.join(class_dir, f), self.label_to_idx[label]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert('RGB' if self.channel_dim == 3 else 'L')
        if self.transform:
            image = self.transform(image)
        return image, label


def main():
    # Try to switch to cuda
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True

    # Data augmentation for training
    train_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(10),
        transforms.Resize(input_dim),
    ])
    # Test and validation transform to match project description
    eval_transform = transforms.Compose([transforms.ToTensor(),transforms.Resize(input_dim),])

    train_dataset = CustomImageDataset('./data_split/training', channel_dim, transform=train_transform)
    val_dataset = CustomImageDataset('./data_split/validation', channel_dim, transform=eval_transform)
    test_dataset = CustomImageDataset('./data_split/testing', channel_dim, transform=eval_transform)

    batchsize = 64
    numworkers = 4
    train_loader = DataLoader(train_dataset, batch_size=batchsize, shuffle=True, num_workers=numworkers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batchsize, shuffle=False, num_workers=numworkers, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batchsize, shuffle=False, num_workers=numworkers, pin_memory=True)

    model = group_1().to(device)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total trainable parameters: {total_params:,}")

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.0005, momentum=0.9)

    num_epochs = 100
    loss_values, val_loss_values = [], []
    best_acc = 0
    best_epoch = 0
    epochs_without_improvement = 0
    patience = 7

    for epoch in range(num_epochs):
        model.train()
        loss_temp = []
        for data, targets in train_loader:
            data, targets = data.to(device, non_blocking=True), targets.to(device, non_blocking=True)
            optimizer.zero_grad()
            outputs = model(data)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            loss_temp.append(loss.item())
        loss_values.append(sum(loss_temp) / len(loss_temp))

        model.eval()
        correct, total = 0, 0
        val_loss_temp = []
        with torch.no_grad():
            for data, targets in val_loader:
                data, targets = data.to(device, non_blocking=True), targets.to(device, non_blocking=True)
                outputs = model(data)
                val_loss_temp.append(criterion(outputs, targets).item())
                _, predicted = torch.max(outputs, dim=1)
                total += targets.size(0)
                correct += (predicted == targets).sum().item()
        val_loss_values.append(sum(val_loss_temp) / len(val_loss_temp))

        val_accuracy = 100 * correct / total
        if val_accuracy > best_acc:
            torch.save(model.state_dict(), 'group_1.pth')
            best_acc = val_accuracy
            best_epoch = epoch + 1
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        print(f"Epoch: {epoch + 1} | Val Accuracy: {val_accuracy:.2f}% | loss: {loss_values[-1]:.4f} | val_loss: {val_loss_values[-1]:.4f}")

        if epochs_without_improvement > patience:
            print(f"Epochs without improvement exceeded {patience}. Early stopping triggered.")
            break

    print(f"Best model saved with a validation accuracy of {best_acc:.2f}% from epoch {best_epoch}")

    # Final hold-out test set
    model.load_state_dict(torch.load('group_1.pth'))
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for data, targets in test_loader:
            data, targets = data.to(device), targets.to(device)
            outputs = model(data)
            _, predicted = torch.max(outputs, dim=1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()
    test_accuracy = 100 * correct / total
    print(f"Hold-out test accuracy: {test_accuracy:.2f}%")

    epochs = range(1, len(loss_values) + 1)
    plt.plot(epochs, loss_values, 'bo', label='Training loss')
    plt.plot(epochs, val_loss_values, 'b', label='Validation loss')
    plt.axvline(x=best_epoch, color='red', linestyle='--', linewidth=1.5, label='Best epoch')
    plt.title(f"Training and validation loss | Params: {total_params:,}\nBest val: {best_acc:.2f}% (Epoch {best_epoch}) | Hold-out test: {test_accuracy:.2f}%")
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig('figure.png')


if __name__ == '__main__':
    main()