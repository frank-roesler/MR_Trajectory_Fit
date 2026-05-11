import os
import random
import shutil
from glob import glob

data_dir = "dcf_generation/train_data/"
files = glob(os.path.join(data_dir, "*.pt"))
random.shuffle(files)

n_files = len(files)
train_split = int(0.8 * n_files)
val_split = int(0.9 * n_files)

splits = {
    "train": files[:train_split],
    "val": files[train_split:val_split],
    "test": files[val_split:]
}

# Create folders and move files
for split_name, split_files in splits.items():
    split_dir = os.path.join(data_dir, split_name)
    os.makedirs(split_dir, exist_ok=True)
    for f in split_files:
        shutil.move(f, os.path.join(split_dir, os.path.basename(f)))

print(f"Split complete: {len(splits['train'])} Train, {len(splits['val'])} Val, {len(splits['test'])} Test.")