import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import time

from group_1 import group_1, input_dim, channel_dim

torch.set_num_threads(4)

class CustomImageDataset(Dataset):
    def __init__(self, root_dir, channel_dim, input_dim, transform=None):
        self.channel_dim = channel_dim
        self.input_dim = input_dim
        self.transform = transform
        self.classes = sorted(os.listdir(root_dir))  # ['normal', 'pneumonia']
        self.label_to_idx = {label: idx for idx, label in enumerate(self.classes)}

        base_resize = transforms.Resize(self.input_dim)

        self.images = []
        self.labels = []
        
        for label in self.classes:
            class_dir = os.path.join(root_dir, label)
            for f in os.listdir(class_dir):
                if f.lower().endswith(('.jpg', '.jpeg')):
                    img_path = os.path.join(class_dir, f)
                    img = Image.open(img_path).convert('RGB' if self.channel_dim == 3 else 'L')
                    img = base_resize(img)
                    self.images.append(img)
                    self.labels.append(self.label_to_idx[label])


    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]
        
        if self.transform:
            image = self.transform(image)

        return image, label


def main():
    # Try to switch to cuda
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True

    print(f"GPU Available: {torch.cuda.is_available()}")
    print(f"Training on device: {device}")

    # Data augmentation for training
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
    ])
    # Test and validation transform to match project description
    eval_transform = transforms.Compose([transforms.ToTensor()])

    train_dataset = CustomImageDataset('./data_split/training', channel_dim, input_dim=input_dim, transform=train_transform)
    #val_dataset = CustomImageDataset('./data_split/validation5', channel_dim, input_dim=input_dim, transform=eval_transform)
    test_dataset = CustomImageDataset('./data_split/testing', channel_dim, input_dim=input_dim, transform=eval_transform)

    batchsize = 64
    numworkers = 2
    train_loader = DataLoader(train_dataset, batch_size=batchsize, shuffle=True, num_workers=numworkers, pin_memory=True)
    #val_loader = DataLoader(val_dataset, batch_size=batchsize, shuffle=False, num_workers=numworkers, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batchsize, shuffle=False, num_workers=numworkers, pin_memory=True)

    print_all_models = False
    print_updates = False
    
    num_epochs = 200
    patience = 10
    over_patience_count = 0

    num_models = 20
    model = []
    for i in range(num_models):
        model.append(group_1().to(device))
        #model[i].load_state_dict(torch.load(f'outputs/models/group_1_model_{i}.pth'))


    total_params = sum(p.numel() for p in model[0].parameters() if p.requires_grad)
    print(f"Total trainable parameters: {total_params:,}")

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = [torch.optim.Adam(model[i].parameters(), lr=0.002, weight_decay=1e-4) for i in range(num_models)]
    scheduler = [torch.optim.lr_scheduler.CosineAnnealingLR(optimizer[i], T_max=num_epochs) for i in range(num_models)]

    loss_values = [[] for i in range(num_models)]
    val_loss_values = [[] for i in range(num_models)]
    val_accuracy = [0.0] * num_models
    test_accuracies = [0.0] * num_models

    best_acc = []
    best_epoch = []
    epochs_without_improvement = []

    for i in range(num_models):
        best_acc.append(0.0)
        best_epoch.append(0)
        epochs_without_improvement.append(0)

    for epoch in range(num_epochs):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        start_time = time.perf_counter()

        for m in model:
            m.train()
        loss_temp = [[] for i in range(num_models)]

        for data, targets in train_loader:
            data, targets = data.to(device, non_blocking=True), targets.to(device, non_blocking=True)
            for i in range(num_models):
                optimizer[i].zero_grad()

                outputs = model[i](data)
                loss = criterion(outputs, targets)

                loss.backward()
                optimizer[i].step()
                loss_temp[i].append(loss.item())

        for i in range(num_models):
            scheduler[i].step
            loss_values[i].append(sum(loss_temp[i]) / len(loss_temp[i]))


        for m in model:
            m.eval()
        correct = [0] * num_models
        total = [0] * num_models
        val_loss_temp = [[] for i in range(num_models)]

        with torch.no_grad():
            for data, targets in train_loader:
                data, targets = data.to(device, non_blocking=True), targets.to(device, non_blocking=True)
                for i in range(num_models):
                    outputs = model[i](data)
                    val_loss_temp[i].append(criterion(outputs, targets).item())
                    _, predicted = torch.max(outputs, dim=1)
                    total[i] += targets.size(0)
                    correct[i] += (predicted == targets).sum().item()

        for i in range(num_models):
            val_loss_values[i].append(sum(val_loss_temp[i]) / len(val_loss_temp[i]))
            val_accuracy[i] = 100 * correct[i] / total[i]

            if val_accuracy[i] >= best_acc[i]:
                torch.save(model[i].state_dict(), f'outputs/models/group_1_model_{i}.pth')
                best_acc[i] = val_accuracy[i]
                best_epoch[i] = epoch + 1
                epochs_without_improvement[i] = 0
                if print_updates:
                    print(f"An improvement has been made to model {i}")
            else:
                epochs_without_improvement[i] += 1

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        end_time = time.perf_counter()

        epoch_best_acc = 0
        epoch_worst_acc = 100
        epoch_avg_acc = 0
        epoch_best_model = 0
        for i in range(num_models):
            if print_all_models:
                print(f"Model: {i} | Val Accuracy: {val_accuracy[i]:.2f}% | loss: {loss_values[i][-1]:.4f} | val_loss: {val_loss_values[i][-1]:.4f}")
            if epoch_best_acc < val_accuracy[i]:
                epoch_best_acc = val_accuracy[i]
                epoch_best_model = i
            if epoch_worst_acc > val_accuracy[i]:
                epoch_worst_acc = val_accuracy[i]
            epoch_avg_acc += val_accuracy[i]
        epoch_avg_acc = epoch_avg_acc / num_models

        if epoch % 5 == 4:
            print(f"Epoch: {epoch + 1} | Acc. high: {epoch_best_acc:.2f}% | Acc. low: {epoch_worst_acc:.2f}% | Acc. avg.: {epoch_avg_acc:.2f}%")

        #print(f"Epoch: {epoch + 1} | Epoch period: {(end_time - start_time):.2f} | Highest accuracy this epoch: {epoch_best_acc:.2f}% by model {epoch_best_model}")

        over_patience_count = 0
        for i in range(num_models):
            if epochs_without_improvement[i] > patience:
                over_patience_count += 1

        if over_patience_count >= num_models:
            print(f"Epochs without improvement exceeded {patience}. Early stopping triggered.")
            break

    print(f"Best epoch for each model saved with the following accuracies:")

    for i in range(num_models):
        print(f"Model {i}: {best_acc[i]:.2f}% from epoch {best_epoch[i]}")

    # Final hold-out test set
    print("Hold-out test of models")
    for i in range(num_models):
        model[i].load_state_dict(torch.load(f'outputs/models/group_1_model_{i}.pth'))
        model[i].eval()
        plot_correct, plot_total = 0, 0
        with torch.no_grad():
            for data, targets in test_loader:
                data, targets = data.to(device), targets.to(device)
                outputs = model[i](data)
                _, predicted = torch.max(outputs, dim=1)
                plot_total += targets.size(0)
                plot_correct += (predicted == targets).sum().item()
        test_accuracies[i] = 100 * plot_correct / plot_total
        print(f"Model {i} test accuracy: {test_accuracies[i]:.2f}%")
    
        epochs = range(1, len(loss_values[i]) + 1)
        plt.plot(epochs, loss_values[i], 'bo', label='Training loss')
        plt.plot(epochs, val_loss_values[i], 'b', label='Validation loss')
        plt.axvline(x=best_epoch[i], color='red', linestyle='--', linewidth=1.5, label='Best epoch')
        plt.title(f"Training and validation loss | Params: {total_params:,}\nBest val: {best_acc[i]:.2f}% (Epoch {best_epoch[i]}) | Hold-out test: {test_accuracies[i]:.2f}%")
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.savefig(f'outputs/graphs/group_1_model_{i}.png')
        plt.close()


if __name__ == '__main__':
    main()