import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from sympy import false
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import numpy as np
import pandas as pd
import os
from torch.optim.lr_scheduler import ReduceLROnPlateau
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
from torch.utils.data import Subset
from Utils import get_loss, get_optimizer, get_metric, set_seed, custom_collate_clip
from Train import train_model
from MLP import MLPHead
import timm
from transformers import AutoTokenizer, AutoModel



# -----------------------------
# Dataset
# -----------------------------
class multimodality(Dataset):
    """
            Initializes the custom dataset for MiniCLIP model.

            Parameters:
            - csv_file (str): Path to the CSV file containing image IDs, captions, and optionally labels.
            - image_root (str): Directory containing the image files.
            - transform (callable, optional): Optional transformation to be applied to the images (e.g., resizing).
            - num_labels (int): Total number of possible labels (for multi-label classification).
            - with_labels (bool): Indicates whether the dataset includes labels. Set to False for test data.
    """

    def __init__(self, csv_file, image_root, transform=None, num_labels=18, with_labels=True):
        self.df = pd.read_csv(csv_file, names=["ImageID", "Caption"], header=None) if not with_labels else pd.read_csv(
            csv_file)
        self.image_root = image_root
        self.transform = transform
        self.num_labels = num_labels
        self.with_labels = with_labels

    def __len__(self):
        """
            Returns the total number of samples in the dataset.
        """
        return len(self.df)

    def __getitem__(self, idx):
        """
            Retrieves the sample at the specified index.

            Parameters:
                - idx (int): Index of the sample to retrieve.

            Returns:
                - If with_labels is True: a tuple (image_tensor, caption, label_tensor)
                - If with_labels is False: a tuple (image_tensor, caption)
        """
        row = self.df.iloc[idx]
        image_path = os.path.join(self.image_root, row['ImageID'])  # Extract the row at the given index
        image = Image.open(image_path).convert("RGB")  # Load and convert the image to RGB format
        if self.transform:
            image = self.transform(image)
        # Extract the caption (text) from the row
        text = row['Caption']

        if self.with_labels:
            # Parse the labels as a string of space-separated integers
            label_str = row['Labels']
            label_list = [int(x) for x in label_str.strip().split() if x.isdigit()]
            # Initialize a zero tensor for multi-label classification
            labels = torch.zeros(self.num_labels, dtype=torch.float)
            # Set the corresponding indices to 1.0 for each present label (indexing from 1)
            for label in label_list:
                if 1 <= label <= self.num_labels:
                    labels[label - 1] = 1.0
            return image, text, labels
        else:
            return image, text


# -----------------------------
# Dataloader preparation
# -----------------------------
def prepare_multimodality_dataloaders(csv_file, image_root, batch_size=16, num_labels=18, train_ratio=0.8):
    # Define image preprocessing and augmentation pipeline
    transform = transforms.Compose([
        transforms.Resize(size=224, interpolation=transforms.InterpolationMode.BICUBIC),  # Resize image
        transforms.CenterCrop(224),                          # Center crop to 224x224
        transforms.RandomHorizontalFlip(),                   # Random horizontal flip
        transforms.ToTensor(),                               # Convert PIL image to tensor
        transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],  # Normalize (CLIP-style)
                             std=[0.26862954, 0.26130258, 0.27577711]),
    ])

    # Load custom multimodal dataset (expects image-text-label triplets)
    dataset = multimodality(csv_file, image_root, transform=transform, num_labels=num_labels)

    # Determine training and validation split sizes
    train_size = int(train_ratio * len(dataset))
    val_size = len(dataset) - train_size

    # Extract all labels for stratified splitting
    labels = []
    for i in range(len(dataset)):
        _, _, y = dataset[i]         # y is a multi-hot encoded vector
        labels.append(y.numpy())
    labels = np.array(labels)

    # Use Multilabel Stratified K-Fold to preserve label distribution in both splits
    mskf = MultilabelStratifiedKFold(n_splits=int(1 / (1 - train_ratio)), shuffle=True, random_state=42)

    # Perform the split and take only the first fold
    for train_idx, val_idx in mskf.split(np.zeros(len(labels)), labels):
        break

    # Create subset datasets using the indices from stratified split
    train_dataset = Subset(dataset, train_idx)
    val_dataset = Subset(dataset, val_idx)

    # Wrap datasets in DataLoader with custom collate function for text batching
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0,
                              collate_fn=custom_collate_clip)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0,
                            collate_fn=custom_collate_clip)

    return train_loader, val_loader


# -----------------------------
# ViT + TinyBERT CLIP-style Dual Encoder
# -----------------------------
class CLIPStyleModel(nn.Module):
    def __init__(self, vit_ckpt_path="vit_small_patch16_224.pth", tinybert_dir="tinybert", proj_dim=312,
                 freeze_backbone=True,dropout_rate=0.3,activation='relu',
                 use_batchnorm=False,use_dropout=True,hidden_dims=[512, 256],init_temperature=0.07):
        super().__init__()
        # Load ViT-Tiny vision encoder (from timm) and its pretrained weights
        self.visual_model = timm.create_model("vit_small_patch16_224", pretrained=False)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        state_dict = torch.load(vit_ckpt_path, map_location=device)
        self.visual_model.load_state_dict(state_dict)
        vision_output_dim = self.visual_model.head.in_features
        self.visual_model.reset_classifier(0) # Remove classification head
        # Load TinyBERT tokenizer and text encoder
        self.tokenizer = AutoTokenizer.from_pretrained(tinybert_dir)
        self.text_model = AutoModel.from_pretrained(tinybert_dir)
        text_output_dim = self.text_model.config.hidden_size

        # Project both modalities to a shared feature space of dimension proj_dim
        self.vision_proj = nn.Linear(vision_output_dim, proj_dim)
        self.text_proj = nn.Linear(text_output_dim, proj_dim)
        # Define a classification head over the concatenated features

        # Add learnable logit scale (temperature)
        self.logit_scale = nn.Parameter(torch.ones([]) * torch.log(torch.tensor(1 / init_temperature)))
        self.classifier = MLPHead(
            input_dim=proj_dim * 2,
            hidden_dims=hidden_dims,
            output_dim=18,
            dropout_rate=dropout_rate,
            activation=activation,
            use_batchnorm=use_batchnorm,
            use_dropout=use_dropout
        )
        # Optionally freeze the pretrained backbone (vision and text encoders)
        if freeze_backbone:
            for param in self.visual_model.parameters():
                param.requires_grad = False
            for param in self.text_model.parameters():
                param.requires_grad = False

    def forward(self, image, text, return_features=False):
        # Forward image through ViT and projection head
        image_feat = self.visual_model(image)
        image_feat = self.vision_proj(image_feat)

        # Tokenize and encode text using TinyBERT
        inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=128)
        inputs = {k: v.to(image.device) for k, v in inputs.items()}
        text_outputs = self.text_model(**inputs)
        text_feat = text_outputs.last_hidden_state[:, 0, :] # Use [CLS] token
        text_feat = self.text_proj(text_feat)


        if return_features:
            # Return normalized features for contrastive learning
            image_feat = nn.functional.normalize(image_feat, dim=1)
            text_feat = nn.functional.normalize(text_feat, dim=1)
            return image_feat, text_feat
        else:
            fused = torch.cat([image_feat, text_feat], dim=1)
            return self.classifier(fused)

# -----------------------------
# CLIP-style contrastive loss
# -----------------------------
def clip_contrastive_loss(image_features, text_features, model):
    # Compute the cosine similarity (dot product) between all image-text pairs
    # Then scale by temperature to control the sharpness of similarity distribution
    logit_scale = model.logit_scale.exp()  # learnable parameter
    logits = image_features @ text_features.T * logit_scale

    # Create ground-truth labels: each image matches only with its corresponding text
    labels = torch.arange(logits.size(0)).to(logits.device)

    # Compute cross-entropy loss for image-to-text direction
    loss_i2t = nn.CrossEntropyLoss()(logits, labels)

    # Compute cross-entropy loss for text-to-image direction
    loss_t2i = nn.CrossEntropyLoss()(logits.T, labels)

    # Return the average of both losses (symmetric contrastive loss)
    return (loss_i2t + loss_t2i) / 2

# -----------------------------
# Training function for CLIP-style model
# -----------------------------
def train_clip_and_classify(train_loader, val_loader, num_labels=18, num_epochs_clip=10,
                            num_epochs_cls=10, lr=1e-4, weight_decay=1e-5, temperature=0.07,
                            threshold=0.5, freeze_backbone=True, clip_patience=3, cls_patience=5,
                            use_asymmetric_loss=False,proj_dim=312,
                            dropout_rate=0.3,activation='relu',
                            use_batchnorm=False,use_dropout=True,optimizer_type='adam',
                            csv_path='training_log.csv',file_name='Clip',hidden_dims=[512, 256],
                            save_classifier_only=False,save_par=False):
    # Set random seed and device
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Initialize the CLIP-style model with projection head and optional dropout, batchnorm
    model = CLIPStyleModel(freeze_backbone=freeze_backbone,proj_dim=proj_dim,
                           dropout_rate=dropout_rate,activation=activation,
                           use_batchnorm=use_batchnorm,use_dropout=use_dropout,
                           hidden_dims=hidden_dims,init_temperature=temperature).to(device)
    # Create optimizer and learning rate scheduler
    optimizer = get_optimizer(model, lr, weight_decay,optimizer_type)
    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=2)
    # Stage 1: Pretrain using CLIP-style contrastive learning
    print("--- Stage 1: CLIP contrastive training ---")
    best_clip_loss = float('inf')
    clip_patience_counter = 0
    for epoch in range(num_epochs_clip):
        model.train()
        total_loss = 0
        for images, texts, _ in train_loader:
            images = images.to(device)
            # Forward pass to obtain image and text features
            image_feat, text_feat = model(images, texts, return_features=True)
            # Compute contrastive loss between image and text features
            loss = clip_contrastive_loss(image_feat, text_feat, model)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        print(f"[Epoch {epoch+1}/{num_epochs_clip}] CLIP Loss: {avg_loss:.4f}")

        if avg_loss < best_clip_loss:
            best_clip_loss = avg_loss
            clip_patience_counter = 0
        else:
            clip_patience_counter += 1
            if clip_patience_counter >= clip_patience:
                print(f"Early stopping triggered at epoch {epoch+1} (CLIP stage)")
                break

    # === Save CLIP model after contrastive training ===
    print("\n--- Stage 2: MLP classification fine-tuning ---")
    criterion = get_loss(use_asymmetric_loss)   # Choose loss function
    metric = get_metric(num_labels, device)  # Evaluation metric, e.g. F1 score
    # Train MLP classification head with early stopping and evaluation
    train_losses, val_losses, train_f1s, val_f1s = train_model(model, train_loader, val_loader, criterion, optimizer,
                                                               scheduler, metric,
                                                               num_epochs=num_epochs_cls, device=device, patience=cls_patience,
                                                               threshold=threshold,
                                                               model_name="Mult", file_name=file_name,
                                                               csv_log_path=csv_path,
                                                               save_classifier_only=save_classifier_only,
                                                               save_par=save_par)
    # Return the trained model and metrics
    return {
        "model": model,
        "train_loss": train_losses,
        "val_loss": val_losses,
        "train_f1": train_f1s,
        "val_f1": val_f1s
    }








