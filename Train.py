import torch
from datetime import datetime
import os
import pandas as pd

def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, metric,
                num_epochs=10, device="cuda", patience=5, threshold=0.5, model_name="ResNet", file_name="model", csv_log_path=None,
                save_classifier_only=False,save_par=False):
    """
    Generic training loop for multi-label classification models (with optional text inputs).

    Args:
        model: PyTorch model
        train_loader: DataLoader for training
        val_loader: DataLoader for validation
        criterion: loss function
        optimizer: optimizer
        scheduler: learning rate scheduler
        metric: torchmetrics metric (e.g., F1Score)
        num_epochs (int): number of training epochs
        device (str): "cuda" or "cpu"
        patience (int): early stopping patience
        threshold (float): sigmoid threshold for multilabel classification
        model_name (str): prefix for saved model filename

    Returns:
        Lists of train_losses, val_losses, train_f1s, val_f1s per epoch
    """

    best_val_f1 = 0
    patience_counter = 0
    train_losses, val_losses = [], []
    train_f1s, val_f1s = [], []
    train_metric = metric
    log_rows = []

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        train_metric.reset()
        if model_name == 'ResNet':
            for images, _, labels in train_loader:
                batch_loss=train_ResNet_epoch(model=model,images=images,labels=labels ,criterion=criterion, optimizer=optimizer,
                             metric=train_metric,device=device,threshold=threshold,mode='train')
                total_loss += batch_loss
        elif model_name == 'Bert':
            for batch in train_loader:
                batch_loss = train_Bert_epoch(model=model, batch=batch, criterion=criterion,
                                          optimizer=optimizer,
                                          metric=train_metric, device=device, threshold=threshold, mode='train')
                total_loss += batch_loss
        elif model_name == 'ViT':
            for images, labels in train_loader:
                batch_loss = train_ViT_epoch(model=model, images=images,
                                              labels=labels,criterion=criterion,optimizer=optimizer,
                                              metric=train_metric, device=device, threshold=threshold, mode='train')
                total_loss += batch_loss
        elif model_name == 'Mult':
            for images, texts, labels in train_loader:
                batch_loss = train_mult_epoch(model=model, images=images, texts=texts,
                                              labels=labels,criterion=criterion,optimizer=optimizer,
                                              metric=train_metric, device=device, threshold=threshold, mode='train')
                total_loss += batch_loss
        elif model_name == 'Clip':
            for images, texts, labels in train_loader:
                batch_loss = train_clip_epoch(model=model, images=images, texts=texts,
                                              labels=labels,criterion=criterion,optimizer=optimizer,
                                              metric=train_metric, device=device, threshold=threshold, mode='train')
                total_loss += batch_loss

        avg_train_loss = total_loss / len(train_loader)
        train_f1 = metric.compute().item()
        train_losses.append(avg_train_loss)
        train_f1s.append(train_f1)

        model.eval()
        val_total_loss = 0
        val_metric = metric
        val_metric.reset()

        with torch.no_grad():
            if model_name == 'ResNet':
                for images, _, labels in val_loader:
                    batch_loss=train_ResNet_epoch(model=model, images=images, labels=labels, criterion=criterion, optimizer=optimizer,
                                 metric=val_metric, device=device, threshold=threshold,
                                 mode='val')
                    val_total_loss += batch_loss
            elif model_name == 'Bert':
                for batch in val_loader:
                    batch_loss = train_Bert_epoch(model=model, batch=batch , criterion=criterion,
                                                    optimizer=optimizer,
                                                    metric=val_metric, device=device, threshold=threshold,
                                                    mode='val')
                    val_total_loss += batch_loss
            elif model_name == 'ViT':
                for images, labels in val_loader:
                    batch_loss = train_ViT_epoch(model=model, images=images, labels=labels, criterion=criterion, optimizer=optimizer,
                                                  metric=train_metric, device=device, threshold=threshold, mode='val')
                    val_total_loss += batch_loss
            elif model_name == 'Mult':
                for images, texts, labels in val_loader:
                    batch_loss = train_mult_epoch(model=model, images=images, texts=texts,
                                                  labels=labels, criterion=criterion, optimizer=optimizer,
                                                  metric=train_metric, device=device, threshold=threshold, mode='val')
                    val_total_loss += batch_loss
            elif model_name == 'Clip':
                for images, texts, labels in val_loader:
                    batch_loss = train_clip_epoch(model=model, images=images, texts=texts,
                                                  labels=labels, criterion=criterion, optimizer=optimizer,
                                                  metric=train_metric, device=device, threshold=threshold, mode='val')
                    val_total_loss += batch_loss
        avg_val_loss = val_total_loss / len(val_loader)
        val_f1 = val_metric.compute().item()
        val_losses.append(avg_val_loss)
        val_f1s.append(val_f1)

        print(f"[Epoch {epoch+1}/{num_epochs}] Train Loss: {avg_train_loss:.4f},Train F1: {train_f1:.4f} | Val Loss: {avg_val_loss:.4f}, Val F1: {val_f1:.4f}")
        # Record
        log_rows.append({
            "run_name": file_name,
            "epoch": epoch + 1,
            "train_loss": avg_train_loss,
            "train_f1": train_f1,
            "val_loss": avg_val_loss,
            "val_f1": val_f1
        })
        scheduler.step(val_f1)

        # Early stopping
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            patience_counter = 0
            if save_par:
                if save_classifier_only and hasattr(model, "classifier"):
                    torch.save({"classifier": model.classifier.state_dict()}, f"{file_name}.pth")
                else:
                    torch.save(model.state_dict(), f"{file_name}.pth")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping triggered at epoch {epoch+1}")
                break
    # Save training/validation log to CSV
    if csv_log_path is not None:
        save_log_to_csv(log_rows, csv_log_path)
    return train_losses, val_losses, train_f1s, val_f1s


def train_ResNet_epoch(model,images,labels, criterion, optimizer, metric,device,threshold, mode='train'):
    images, labels = images.to(device), labels.to(device)
    if mode == 'train':
        optimizer.zero_grad()

    outputs = model(images)
    loss = criterion(outputs, labels)

    if mode == 'train':
        loss.backward()
        optimizer.step()

    preds = (torch.sigmoid(outputs) > threshold).float()
    metric.update(preds, labels)
    return loss.item()

def train_ViT_epoch(model,images,labels, criterion, optimizer, metric,device,threshold, mode='train'):
    images, labels = images.to(device), labels.to(device)
    if mode == 'train':
        optimizer.zero_grad()

    outputs = model(images)
    loss = criterion(outputs, labels)

    if mode == 'train':
        loss.backward()
        optimizer.step()

    preds = (torch.sigmoid(outputs) > threshold).float()
    metric.update(preds, labels)
    return loss.item()

def train_Bert_epoch(model,batch, criterion, optimizer, metric,device,threshold, mode='train'):
    input_ids = batch['input_ids'].to(device)
    attention_mask = batch['attention_mask'].to(device)
    labels = batch['labels'].to(device)

    if mode == 'train':
        optimizer.zero_grad()

    outputs = model(input_ids, attention_mask)
    loss = criterion(outputs, labels)

    if mode == 'train':
        loss.backward()
        optimizer.step()

    preds = (torch.sigmoid(outputs) > threshold).float()
    metric.update(preds, labels)
    return loss.item()

def train_mult_epoch(model, images, texts, labels , criterion, optimizer, metric, device, threshold, mode='train'):
    images, labels = images.to(device), labels.to(device)

    if mode == 'train':
        optimizer.zero_grad()

    outputs = model(images, texts)
    loss = criterion(outputs, labels)
    if mode == 'train':
        loss.backward()
        optimizer.step()

    preds = (torch.sigmoid(outputs) > threshold).float()
    metric.update(preds, labels)
    return loss.item()

def train_clip_epoch(model, images, texts, labels , criterion, optimizer, metric, device, threshold, mode='train'):
    images, labels = images.to(device), labels.to(device)

    if mode == 'train':
        optimizer.zero_grad()

    outputs = model(images, texts)
    loss = criterion(outputs, labels)
    if mode == 'train':
        loss.backward()
        optimizer.step()

    preds = (torch.sigmoid(outputs) > threshold).float()
    metric.update(preds, labels)
    return loss.item()

def save_log_to_csv(log_rows, csv_log_path):
    """
    Save training/validation log to a CSV file, with timestamp.

    Args:
        log_rows (list of dict): list of training log data
        csv_log_path (str): file path to save log CSV
    """
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    for row in log_rows:
        row["timestamp"] = timestamp

    if os.path.exists(csv_log_path):
        existing_df = pd.read_csv(csv_log_path)
        new_df = pd.DataFrame(log_rows)
        combined_df = pd.concat([existing_df, new_df], ignore_index=True)
        combined_df.to_csv(csv_log_path, index=False)
    else:
        pd.DataFrame(log_rows).to_csv(csv_log_path, index=False)