import torch
import torch.nn as nn
from torchmetrics import F1Score
import random
import numpy as np

# Define the optimizer (Adam)
def get_optimizer(model, lr=1e-3, weight_decay=0.0, optimizer_type="adam"):
    """
    Returns a configured optimizer for the given model.

    Args:
        model (torch.nn.Module): The model whose parameters will be optimized.
        lr (float): Learning rate for the optimizer.
        weight_decay (float): Weight decay (L2 penalty) for regularization.
        optimizer_type (str): Type of optimizer to use. Must be either "adam" or "adamw".

    Returns:
        torch.optim.Optimizer: Configured optimizer instance.
    """
    params = filter(lambda p: p.requires_grad, model.parameters())

    if optimizer_type.lower() == "adam":
        return torch.optim.Adam(params, lr=lr, weight_decay=weight_decay)
    elif optimizer_type.lower() == "adamw":
        return torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
    else:
        raise ValueError(f"Unsupported optimizer_type: {optimizer_type}. Use 'adam' or 'adamw'.")


def asymmetric_loss_with_logits(pred, targets, gamma_neg=4, clip=0.05):
    """
    Asymmetric loss for multi-label classification.
    pred: logits
    target: binary labels
    """
    probs = torch.sigmoid(pred)
    pos_loss = targets * torch.log(probs + 1e-8)
    neg_loss = (1 - targets) * torch.log(1 - probs + 1e-8)
    loss = - (4 * pos_loss + neg_loss)  # gamma adjustment
    return loss.mean()

# Define the loss function (binary cross-entropy with logits)
def get_loss(use_asymmetric=False):
    if use_asymmetric:
        return asymmetric_loss_with_logits
    else:
        return nn.BCEWithLogitsLoss()


def get_metric(num_labels=18, device="cuda"):
    """
    Returns a multi-label F1 score metric instance for evaluation.

    Args:
        num_labels (int): The number of labels/classes in the multi-label classification task.
        device (str): The device on which to place the metric (e.g., 'cuda' or 'cpu').

    Returns:
        torchmetrics.F1Score: An F1Score metric object configured for multi-label classification.
    """
    # Create and return the F1Score metric for multi-label classification
    return F1Score(task="multilabel", num_labels=num_labels).to(device)


# Set random seed for reproducibility
def set_seed(seed=42):
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.

    Args:
        seed (int): The seed value to use. Default is 42.

    This function ensures that:
    - Randomness in Python's `random` module is controlled
    - NumPy-generated randomness is consistent
    - PyTorch CPU and GPU randomness is reproducible
    - CUDA operations are deterministic to the extent possible
    """
    # Set seed for PyTorch (CPU)
    torch.manual_seed(seed)

    # Set seed for all CUDA devices (if using GPU)
    torch.cuda.manual_seed_all(seed)

    # Set seed for NumPy random number generator
    np.random.seed(seed)

    # Set seed for Python's built-in random module
    random.seed(seed)

    # Ensure deterministic behavior in CUDA convolution operations
    torch.backends.cudnn.deterministic = True

    # Disable CuDNN benchmarking to avoid non-deterministic behavior
    torch.backends.cudnn.benchmark = False

# Custom collate function for DataLoader
# Groups a batch of (image, input_ids, attention_mask, labels) tuples into batched tensors
# Useful for multimodal tasks (image + text input)
def custom_collate(batch):
    # Unpack each field from tuples in the batch
    images, input_ids, attention_mask, labels = zip(*batch)
    return (
        torch.stack(images), # shape: (batch_size, C, H, W)
        torch.stack(input_ids), # shape: (batch_size, seq_len)
        torch.stack(attention_mask),  # shape: (batch_size, seq_len)
        torch.stack(labels) # shape: (batch_size, num_labels)
    )


def custom_collate_clip(batch):
    images, texts, labels = zip(*batch)
    return torch.stack(images), list(texts), torch.stack(labels)