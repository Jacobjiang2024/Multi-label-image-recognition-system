# import library
import pandas as pd
import os
from PIL import Image
import torch
import torchvision.transforms.v2 as v2
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.preprocessing import MultiLabelBinarizer
from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from collections import Counter
import csv

# Custom dataset class for handling image-text-label data
class ImageTextData(Dataset):
    def __init__(self, data, transform=True, size=None, sample_mode='random', aug_type="default", resize_shape=(336, 336)):
        # File paths for raw and cleaned CSV
        clean_file = f'COMP5329S1A2Dataset/{data}.csv'

        # Load and sort the cleaned CSV by numeric image ID
        self.df = pd.read_csv(clean_file, quotechar='"')

        # Subsample data if size is specified
        self.resize_shape = resize_shape
        if size is not None:
            if sample_mode == "random": # default to label-aware sampling
                #self.df = sample_with_all_labels(self.df, target_size=size)
                self.df = self.df.sample(n=size, random_state=42).reset_index(drop=True)
            else:
                self.df = self.df.iloc[:size].reset_index(drop=True)

        self.size = len(self.df)
        self.transform = transform

        # augmentation presets to test different types
        self.aug_type = aug_type
        self.image_transform = self.build_transform(aug_type)

        # Load data
        self.images = self.process_image()
        self.texts = self.process_text()
        self.labels = self.process_label()

    # Build transform pipeline based on augmentation type
    def build_transform(self, aug_type):
        base = [v2.ToImage()]

        resize_op = v2.Resize(self.resize_shape)
        if aug_type == "default":
            base += [
                v2.RandomPhotometricDistort(p=1),
                v2.RandomAdjustSharpness(sharpness_factor=2, p=0.5),
                v2.RandomRotation(degrees=15),
                v2.RandomHorizontalFlip(p=0.5),
                resize_op,
            ]
        elif aug_type == "elastic":
            base += [
                v2.ElasticTransform(alpha=250.0),
                v2.RandomHorizontalFlip(p=0.5),
                resize_op,
            ]
        elif aug_type == "coloronly":
            base += [
                v2.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5),
                v2.RandomHorizontalFlip(p=0.5),
                resize_op,
            ]
        elif aug_type == "minimal":
            base += [
                v2.RandomHorizontalFlip(p=0.5),
                resize_op,
            ]
        else:
            # no augmentation fallback
            base += [resize_op,
                     v2.ToImage(), # Convert PIL to Tensor
                     v2.ToDtype(torch.float32, scale=True)] # uint8 → float32


        base.append(v2.Normalize(mean=(0.485, 0.456, 0.406),
                         std=(0.229, 0.224, 0.225)))
        return v2.Compose(base)

    # Clean and write new CSV from raw input file


    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        text = self.texts[idx]
        label = self.labels[idx]

        if self.transform:
            image = self.image_transform(image)

        return image, text, label

    def process_image(self):
        paths = self.df["ImageID"].tolist()
        return [Image.open(f"COMP5329S1A2Dataset/data/{p}").convert("RGB") for p in paths]

    def process_text(self):
        return self.df["Caption"].tolist()

    def process_label(self):
        label_strs = self.df["Labels"].astype(str).tolist()
        # Parse all unique class labels
        all_labels = set()
        for s in label_strs:
            for x in s.split():
                if x.isdigit():
                    all_labels.add(int(x))

        all_labels = sorted(list(all_labels)) # Ensure order consistency
        label2idx = {l: i for i, l in enumerate(all_labels)}
        num_classes = len(all_labels)

        # Convert each row to one-hot Tensor
        one_hot_list = []
        for s in label_strs:
            vec = [0] * num_classes
            for x in s.split():
                if x.isdigit():
                    vec[label2idx[int(x)]] = 1
            one_hot_list.append(torch.tensor(vec, dtype=torch.float))

        return one_hot_list


def clean_multilabel_csv(src_file, dst_file):
    print(f"Cleaning {src_file} to {dst_file} ...")
    with open(src_file, encoding='utf-8') as fin, open(dst_file, "w", encoding='utf-8') as fout:
        for line in fin:
            if not line.strip():
                continue
            cols = line.strip().split(',')
            if len(cols) < 3:
                continue
            imgid = cols[0]
            labels = cols[1]
            caption = ",".join(cols[2:]).strip()

            if '"' in caption:
                caption = caption.replace('"', '""')
            if ',' in caption or '"' in caption:
                caption = f'"{caption}"'

            fout.write(f"{imgid},{labels},{caption}\n")

# Unified split function for both ResNet and BERT with consistent indices

def split_dataset(image_dataset, batch_size=32, test_size=0.2, seed=42, model_type="resnet", tokenizer=None, max_len=128):
    indices = list(range(len(image_dataset)))
    train_idx, val_idx = train_test_split(indices, test_size=test_size, random_state=seed)

    if model_type == "resnet":
        train_dataset = Subset(image_dataset, train_idx)
        val_dataset = Subset(image_dataset, val_idx)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        return train_loader, val_loader
    else:
        raise ValueError("model_type must be either 'bert' or 'resnet'")

# Multi-label stratified split using iterstrat
# returns train_loader, val_loader (just like before)
def stratified_split_dataset(dataset, batch_size=100, test_size=0.2, seed=42):
    """
    Use iterstrat to delimit ImageTextData objects to avoid duplicate loading of image text.
    """
    # Convert labels to multi-hot encoding
    y = dataset.df["Labels"].astype(str).apply(lambda s: [int(x) for x in s.split() if x.isdigit()])
    mlb = MultiLabelBinarizer()
    y_bin = mlb.fit_transform(y)

    # stratified split index
    splitter = MultilabelStratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    X = dataset.df.index.to_numpy()
    train_idx, val_idx = next(splitter.split(X, y_bin))

    # Create a new copy of the dataset (to avoid duplicate loading)
    def copy_subset(indices):
        new_ds = ImageTextData(data="train", transform=dataset.transform,
                               aug_type=dataset.aug_type, resize_shape=dataset.resize_shape)
        new_ds.df = dataset.df.iloc[indices].reset_index(drop=True)
        new_ds.images = [dataset.images[i] for i in indices]
        new_ds.texts = [dataset.texts[i] for i in indices]
        new_ds.labels = [dataset.labels[i] for i in indices]
        return new_ds

    train_ds = copy_subset(train_idx)
    val_ds = copy_subset(val_idx)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader


# Check the number of tags
def check_label_coverage(df, label_col="Labels"):
    label_strs = df[label_col].astype(str).tolist()
    label_set = set()
    for s in label_strs:
        label_set.update(int(l) for l in s.split() if l.isdigit())
    print(f" Number of covered labels: {len(label_set)} -> {sorted(label_set)}")

def clean_test_csv(src_file, dst_file):
    print(f"Cleaning {src_file} → {dst_file}")
    with open(src_file, "r", encoding="utf-8") as fin, open(dst_file, "w", encoding="utf-8", newline='') as fout:
        reader = csv.reader(fin)
        writer = csv.writer(fout, quoting=csv.QUOTE_ALL)

        for row in reader:
            if not row:
                continue
            image_id = row[0].strip()
            caption = ",".join(row[1:]).strip().replace('"', '""')  
            writer.writerow([image_id, caption])

def plot_label_distribution_from_dataset(dataset, label_col="Labels"):
    """
    Visualizing label distribution in ImageTextData (multi-label task)
    """
    label_strs = dataset.df[label_col].astype(str).tolist()

    # Count the number of occurrences of each class
    label_counts = Counter()
    for s in label_strs:
        for label in s.split():
            if label.isdigit():
                label_counts[int(label)] += 1

    labels = sorted(label_counts.keys())
    counts = [label_counts[l] for l in labels]

    # Drawing
    plt.figure(figsize=(10, 5))
    plt.bar(labels, counts, color='skyblue')
    plt.xlabel('Label')
    plt.ylabel('Count')
    plt.title('Label Distribution in Subsampled Dataset')
    plt.xticks(labels)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()

# ========== Debug/Test Code ==========
if __name__ == '__main__':

  print("test")

