import torch
from torch.utils.data import DataLoader, ConcatDataset
from torchvision import transforms
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import shutil

from group_1 import group_1, input_dim, channel_dim
from train import CustomImageDataset


def clear_folder(folder):
    if os.path.exists(folder):
        shutil.rmtree(folder)
    os.makedirs(folder, exist_ok=True)


def evaluate_and_save_misclassified(model, loader, label_names, device, output_dir='outputs/misclassified'):
    clear_folder(output_dir)
    model.eval()

    correct = 0
    total = 0
    misclassified_count = 0

    with torch.no_grad():
        for data, targets in loader:
            data = data.to(device)
            targets_device = targets.to(device)

            outputs = model(data)
            _, predicted = torch.max(outputs, dim=1)

            correct += (predicted == targets_device).sum().item()
            total += targets.size(0)

            wrong_mask = (predicted != targets_device).cpu()

            for img, true_label, pred_label in zip(
                data[wrong_mask].cpu(), targets[wrong_mask], predicted[wrong_mask].cpu()
            ):
                true_name = label_names[true_label.item()]
                pred_name = label_names[pred_label.item()]

                plt.imshow(img.squeeze().numpy(), cmap='gray')
                plt.title(f'True: {true_name} | Predicted: {pred_name}')
                plt.axis('off')
                plt.savefig(f'{output_dir}/true_{true_name}_pred_{pred_name}_{misclassified_count}.png')
                plt.clf()
                misclassified_count += 1

    accuracy = 100 * correct / total
    print(f"Accuracy: {accuracy:.2f}% ({correct}/{total})")
    print(f"Saved {misclassified_count} misclassified images to '{output_dir}'")


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    eval_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize(input_dim),
    ])

    train_ds = CustomImageDataset('./data_split/training', channel_dim, transform=eval_transform)
    val_ds = CustomImageDataset('./data_split/validation', channel_dim, transform=eval_transform)
    test_ds = CustomImageDataset('./data_split/testing', channel_dim, transform=eval_transform)

    full_dataset = ConcatDataset([train_ds, val_ds, test_ds])
    full_loader = DataLoader(full_dataset, batch_size=64, shuffle=False)

    # label_to_idx is the same across all three since folder names match
    label_names = {idx: label for label, idx in train_ds.label_to_idx.items()}

    model = group_1()
    model.load_state_dict(torch.load('group_1.pth', map_location=device))
    model = model.to(device)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total trainable parameters: {total_params:,}")

    evaluate_and_save_misclassified(model, full_loader, label_names, device)


if __name__ == '__main__':
    main()