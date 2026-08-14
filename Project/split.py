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
SPLITS = {'training0': 0.1, 'training1': 0.1, 'training2': 0.1, 'training3': 0.1, 'training4': 0.1, 'training5': 0.1, 'training6': 0.1, 'training7': 0.1, 'training8': 0.1, 'testing': 0.1}

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
    n_train0 = int(n * SPLITS['training0'])
    n_train1 = n_train0 + int(n * SPLITS['training1'])
    n_train2 = n_train1 + int(n * SPLITS['training2'])
    n_train3 = n_train2 + int(n * SPLITS['training3'])
    n_train4 = n_train3 + int(n * SPLITS['training4'])
    n_train5 = n_train4 + int(n * SPLITS['training5'])
    n_train6 = n_train5 + int(n * SPLITS['training6'])
    n_train7 = n_train6 + int(n * SPLITS['training7'])
    n_train8 = n_train7 + int(n * SPLITS['training8'])

    split_files = {
        'training0': files[:n_train0],
        'training1': files[n_train0:n_train1],
        'training2': files[n_train1:n_train2],
        'training3': files[n_train2:n_train3],
        'training4': files[n_train3:n_train4],
        'training5': files[n_train4:n_train5],
        'training6': files[n_train5:n_train6],
        'training7': files[n_train6:n_train7],
        'training8': files[n_train7:n_train8],
        'testing': files[n_train8:],
    }

    for split, split_list in split_files.items():
        for f in split_list:
            shutil.copy2(os.path.join(SOURCE_DIR, f), os.path.join(OUTPUT_DIR, split, label, f))

    #print(f"{label}: {n_train} train / {n_val} val / {n - n_train - n_val} test")