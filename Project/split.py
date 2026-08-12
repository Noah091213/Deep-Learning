import os
import random
import shutil

# This program is only meant to be ran once to create the data_split folder

random.seed(42)  # reproducible split

SOURCE_DIR = './data'
OUTPUT_DIR = './data_split'

# 80/10/10 split with 2200 images total, 220 per val/test should
# be plenty to reliably estimate accuracy against the 95% threshold,
# while keeping most of the data for training the model.
SPLITS = {'training': 0.8, 'validation': 0.1, 'testing': 0.1}

filenames = [f for f in os.listdir(SOURCE_DIR) if f.lower().endswith(('.jpg', '.jpeg'))]
random.shuffle(filenames)

# group filenames by label parsed from filename
by_label = {'normal': [], 'pneumonia': []}
for f in filenames:
    label = os.path.splitext(f)[0].rsplit('_', 1)[-1]
    by_label[label].append(f)

for split in SPLITS:
    for label in by_label:
        os.makedirs(os.path.join(OUTPUT_DIR, split, label), exist_ok=True)

for label, files in by_label.items():
    n = len(files)
    n_train = int(n * SPLITS['training'])
    n_val = int(n * SPLITS['validation'])

    split_files = {
        'training': files[:n_train],
        'validation': files[n_train:n_train + n_val],
        'testing': files[n_train + n_val:],
    }

    for split, split_list in split_files.items():
        for f in split_list:
            shutil.copy2(os.path.join(SOURCE_DIR, f), os.path.join(OUTPUT_DIR, split, label, f))

    print(f"{label}: {n_train} train / {n_val} val / {n - n_train - n_val} test")