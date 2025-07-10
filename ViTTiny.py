import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
from torch.utils.data import Dataset, DataLoader, Subset
import pandas as pd
import numpy as np
import os
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
from torch.optim.lr_scheduler import ReduceLROnPlateau
from Utils import get_loss, get_optimizer, get_metric, set_seed, custom_collate_clip
from Train import train_model
from MLP import MLPHead
import timm

# -----------------------------
# ViT-Tiny-only Image Classifier
# -----------------------------
class ViTTinyClassifier(nn.Module):
    def __init__(self, vit_ckpt_path="vit_small_patch16_224.pt", num_labels=18, dropout=0.1, freeze_backbone=True):
        super().__init__()
        self.visual_model = timm.create_model("vit_small_patch16_224", pretrained=False)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        state_dict = torch.load(vit_ckpt_path, map_location=device)
        self.visual_model.load_state_dict(state_dict)
        vision_output_dim = self.visual_model.head.in_features
        self.visual_model.reset_classifier(0)
        self.vision_proj = nn.Linear(vision_output_dim, 312)

        if freeze_backbone:
            for param in self.visual_model.parameters():
                param.requires_grad = False

        self.classifier = MLPHead(
            input_dim=312,
            hidden_dims=[256, 128],
            output_dim=num_labels,
            dropout_rate=dropout,
            activation='relu',
            use_batchnorm=False,
            use_dropout=True
        )

    def forward(self, image):
        image_feat = self.visual_model(image)
        image_feat = self.vision_proj(image_feat)
        return self.classifier(image_feat)

# -----------------------------
# ViT-only Dataset
# -----------------------------
class ViTOnlyDataset(Dataset):
    def __init__(self, csv_file, image_root, transform=None, num_labels=18):
        self.df = pd.read_csv(csv_file)
        self.image_root = image_root
        self.transform = transform
        self.num_labels = num_labels

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image_path = os.path.join(self.image_root, row['ImageID'])
        image = Image.open(image_path).convert("RGB")
        if self.transform:
            image = self.transform(image)

        label_str = row['Labels']
        label_list = [int(x) for x in label_str.strip().split() if x.isdigit()]
        labels = torch.zeros(self.num_labels, dtype=torch.float)
        for label in label_list:
            if 1 <= label <= self.num_labels:
                labels[label - 1] = 1.0
        return image, labels

# -----------------------------
# Prepare DataLoader
# -----------------------------
def prepare_vitonly_dataloaders(csv_file, image_root, batch_size=16, num_labels=18, train_ratio=0.8):
    transform = transforms.Compose([
        transforms.Resize(size=224, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                             std=[0.26862954, 0.26130258, 0.27577711]),
    ])

    dataset = ViTOnlyDataset(csv_file, image_root, transform=transform, num_labels=num_labels)

    labels = []
    for i in range(len(dataset)):
        _, y = dataset[i]
        labels.append(y.numpy())
    labels = np.array(labels)

    mskf = MultilabelStratifiedKFold(n_splits=int(1 / (1 - train_ratio)), shuffle=True, random_state=42)
    for train_idx, val_idx in mskf.split(np.zeros(len(labels)), labels):
        break

    train_dataset = Subset(dataset, train_idx)
    val_dataset = Subset(dataset, val_idx)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    return train_loader, val_loader

# -----------------------------
# Training function
# -----------------------------
def train_vitonly(train_loader, val_loader, num_labels=18, num_epochs=10, lr=1e-4, weight_decay=1e-5, dropout_rate=0.1,
                  seed=42, patience=5, threshold=0.5, use_asymmetric_loss=False,csv_path='training_log.csv'):
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ViTTinyClassifier(num_labels=num_labels, dropout=dropout_rate).to(device)
    criterion = get_loss(use_asymmetric_loss)
    optimizer = get_optimizer(model, lr, weight_decay)
    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=2)
    metric = get_metric(num_labels, device)

    train_losses, val_losses, train_f1s, val_f1s = train_model(model, train_loader, val_loader, criterion, optimizer,
                                                               scheduler, metric,
                                                               num_epochs=num_epochs, device=device, patience=patience,
                                                               threshold=threshold,
                                                               model_name="ViT", file_name='ViTTiny',
                                                               csv_log_path=csv_path,
                                                               save_classifier_only=False)

    return {
        "model": model,
        "train_loss": train_losses,
        "val_loss": val_losses,
        "train_f1": train_f1s,
        "val_f1": val_f1s
    }
