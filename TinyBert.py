import torch
from torch.utils.data import Dataset
from torch.optim.lr_scheduler import ReduceLROnPlateau
from transformers import AutoTokenizer, AutoModel
from Utils import get_loss, get_optimizer, get_metric, set_seed
from Train import train_model
from MLP import MLPHead
from torch.utils.data import DataLoader
import pandas as pd
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
import numpy as np
from torch.utils.data import Subset
# -----------------------------
# Custom TinyBERT-based model
# -----------------------------
class TinyBERTWithHead(torch.nn.Module):
    def __init__(self, model_path="tinybert", output_dim=18, dropout_rate=0.1, freeze_base=True):
        super().__init__()
        self.bert = AutoModel.from_pretrained(model_path)
        hidden_size = self.bert.config.hidden_size
        self.proj = torch.nn.Linear(hidden_size, 312)
        self.classifier = MLPHead(
            input_dim=312,
            hidden_dims=[256, 128],
            output_dim=output_dim,
            dropout_rate=dropout_rate,
            activation='relu',
            use_batchnorm=False,
            use_dropout=True
        )

        if freeze_base:
            for param in self.bert.parameters():
                param.requires_grad = False

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls_token = outputs.last_hidden_state[:, 0, :]  # CLS token
        x = self.proj(cls_token)
        return self.classifier(x)

# -----------------------------
# Dataset for TinyBERT
# -----------------------------
class TinyBERTCaptionDataset(Dataset):
    def __init__(self, captions, labels, tokenizer, max_len=128):
        self.captions = captions
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.captions)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.captions[idx],
            padding='max_length',
            truncation=True,
            max_length=self.max_len,
            return_tensors='pt'
        )
        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'labels': torch.tensor(self.labels[idx], dtype=torch.float)
        }


def prepare_tinybert_dataloaders_from_csv(csv_path, tokenizer_path='tinybert', num_labels=18, batch_size=16,
                                          train_ratio=0.8):
    df = pd.read_csv(csv_path)
    captions = df['Caption'].tolist()

    labels = []
    for label_str in df['Labels']:
        label_vec = [0.0] * num_labels
        for l in label_str.strip().split():
            if l.isdigit() and 1 <= int(l) <= num_labels:
                label_vec[int(l) - 1] = 1.0
        labels.append(label_vec)

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    dataset = TinyBERTCaptionDataset(captions, labels, tokenizer)

    # Stratified split
    mskf = MultilabelStratifiedKFold(n_splits=int(1 / (1 - train_ratio)), shuffle=True, random_state=42)
    for train_idx, val_idx in mskf.split(np.zeros(len(labels)), np.array(labels)):
        break

    train_loader = DataLoader(Subset(dataset, train_idx), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(Subset(dataset, val_idx), batch_size=batch_size, shuffle=False)
    return train_loader, val_loader



# -----------------------------
# Training function
# -----------------------------
def train_tinybert_caption_classifier(train_loader, val_loader, num_labels=18, num_epochs=10, lr=1e-4, weight_decay=0.01, seed=42, patience=5, use_asymmetric_loss=False, threshold=0.5):
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = TinyBERTWithHead(
        model_path="tinybert",
        output_dim=num_labels,
        dropout_rate=0.1,
        freeze_base=True
    ).to(device)

    optimizer = get_optimizer(model, lr, weight_decay)
    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=2)
    criterion = get_loss(use_asymmetric=use_asymmetric_loss)
    metric = get_metric(num_labels=num_labels, device=device)

    train_losses, val_losses, train_f1s, val_f1s = train_model(model, train_loader, val_loader, criterion, optimizer,
                                                               scheduler, metric,
                                                               num_epochs=num_epochs, device=device, patience=patience,
                                                               threshold=threshold,
                                                               model_name="Bert", file_name='tinybert',
                                                               csv_log_path='training_log.csv',
                                                               save_classifier_only=True)

    return {
        "model": model,
        "train_loss": train_losses,
        "val_loss": val_losses,
        "train_f1": train_f1s,
        "val_f1": val_f1s
    }
