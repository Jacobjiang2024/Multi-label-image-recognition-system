import torch
import torch.nn as nn
import torchvision.models as models
from torch.optim.lr_scheduler import ReduceLROnPlateau
from Utils import get_loss,get_optimizer,get_metric,set_seed
from Train import train_model


# Define a custom ResNet-based model for multi-label classification
class MultiLabelResNet(nn.Module):
    def __init__(self, num_labels=18, backbone="resnet18", dropout_rate=0.5, freeze_backbone=False):
        super(MultiLabelResNet, self).__init__()

        # Load the specified backbone (resnet18 or resnet34) with pretrained weights
        if backbone == "resnet18":
            self.backbone = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        elif backbone == "resnet34":
            self.backbone = models.resnet34(weights=models.ResNet34_Weights.DEFAULT)
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")

        # Optionally freeze backbone parameters
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
            print("Backbone frozen: True")

        # Replace the classification head with a new one for multi-label output
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(in_features, num_labels)
        )

    def forward(self, x):
        return self.backbone(x)


"""
    Train a multi-label image classification model based on ResNet with early stopping and F1 evaluation.

    Parameters:
    - train_loader (DataLoader): Dataloader for training set.
    - val_loader (DataLoader): Dataloader for validation set.
    - num_labels (int): Number of output classes (multi-label setting).
    - num_epochs (int): Total number of training epochs.
    - backbone (str): Backbone model to use: "resnet18" or "resnet34".
    - lr (float): Learning rate.
    - weight_decay (float): Weight decay (L2 regularization).
    - seed (int): Random seed for deterministic training.
    - patience (int): Number of epochs with no improvement after which training will be stopped.
    - dropout_rate (float): Dropout rate used before final classification layer.
    - freeze_backbone (bool): If True, freezes the convolutional backbone during training.
    - use_asymmetric_loss (bool): If True, use asymmetric loss to address label imbalance; 
                                  otherwise use standard BCEWithLogitsLoss.
"""

def train_ResNet(train_loader, val_loader, num_labels=18, num_epochs=10, backbone="resnet18", lr=1e-4, weight_decay=0.0, seed=42, patience=5, dropout_rate=0.5, freeze_backbone=False,use_asymmetric_loss=False,threshold=0.5):
    set_seed(seed)  # Ensure reproducibility
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = MultiLabelResNet(num_labels=num_labels, backbone=backbone, dropout_rate=dropout_rate, freeze_backbone=freeze_backbone).to(device)
    criterion = get_loss(use_asymmetric=use_asymmetric_loss)
    optimizer = get_optimizer(model, lr, weight_decay)
    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=2)
    metric = get_metric(num_labels=num_labels, device=device)

    train_losses,val_losses,train_f1s,val_f1s=train_model(model,train_loader,val_loader,criterion,optimizer,scheduler,metric,
                                                          num_epochs=num_epochs,device=device,patience=patience,threshold=threshold,
                                                          model_name="ResNet",file_name=backbone, csv_log_path='training_log.csv')

    return {
        "model": model,
        "train_loss": train_losses,
        "val_loss": val_losses,
        "train_f1": train_f1s,
        "val_f1": val_f1s
    }
# Training and validation function with EarlyStopping and scheduler
# def train_ResNet(train_loader, val_loader, num_labels=18, num_epochs=10, backbone="resnet18", lr=1e-4, weight_decay=0.0, seed=42, patience=5, dropout_rate=0.5, freeze_backbone=False,use_asymmetric_loss=False):
#     set_seed(seed)  # Ensure reproducibility
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#
#     model = MultiLabelResNet(num_labels=num_labels, backbone=backbone, dropout_rate=dropout_rate, freeze_backbone=freeze_backbone).to(device)
#     criterion = get_loss(use_asymmetric=use_asymmetric_loss)
#     optimizer = get_optimizer(model, lr, weight_decay)
#     scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
#     metric = get_metric(num_labels=num_labels, device=device)
#
#     best_val_f1 = 0
#     patience_counter = 0
#     train_losses, val_losses = [], []
#     train_f1s, val_f1s = [], []
#
#     for epoch in range(num_epochs):
#         model.train()
#         total_loss = 0
#         metric.reset()
#
#         for images, _, labels in train_loader:
#             images, labels = images.to(device), labels.to(device)
#
#             optimizer.zero_grad()
#             outputs = model(images)
#             loss = criterion(outputs, labels)
#             loss.backward()
#             optimizer.step()
#
#             total_loss += loss.item()
#             preds = torch.sigmoid(outputs)
#             metric.update(preds, labels)
#
#         avg_train_loss = total_loss / len(train_loader)
#         train_f1 = metric.compute().item()
#
#         model.eval()
#         total_val_loss = 0
#         metric.reset()
#
#         with torch.no_grad():
#             for images, _, labels in val_loader:
#                 images, labels = images.to(device), labels.to(device)
#                 outputs = model(images)
#                 loss = criterion(outputs, labels)
#                 total_val_loss += loss.item()
#                 preds = torch.sigmoid(outputs)
#                 metric.update(preds, labels)
#
#         avg_val_loss = total_val_loss / len(val_loader)
#         val_f1 = metric.compute().item()
#
#         train_losses.append(avg_train_loss)
#         val_losses.append(avg_val_loss)
#         train_f1s.append(train_f1)
#         val_f1s.append(val_f1)
#
#         print(f"Epoch {epoch+1}/{num_epochs} | Train Loss: {avg_train_loss:.4f}, F1: {train_f1:.4f} | Val Loss: {avg_val_loss:.4f}, F1: {val_f1:.4f}")
#
#         scheduler.step()
#
#         # Early stopping check
#         if val_f1 > best_val_f1:
#             best_val_f1 = val_f1
#             patience_counter = 0
#             # Keep the parameters of the whole model
#             torch.save(model.state_dict(), f"best_{backbone}_model.pt")
#         else:
#             patience_counter += 1
#             if patience_counter >= patience:
#                 print("Early stopping triggered.")
#                 break
#
#     return {
#         "model": model,
#         "train_loss": train_losses,
#         "val_loss": val_losses,
#         "train_f1": train_f1s,
#         "val_f1": val_f1s
#     }
