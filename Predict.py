import torch
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
from Multimodality import CLIPStyleModel
from PIL import Image
import os
from torchvision import transforms


# Custom dataset for inference
class InferenceDataset(torch.utils.data.Dataset):
    def __init__(self, csv_file, image_root, transform):
        self.df = pd.read_csv(csv_file)
        self.image_root = image_root
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image_path = os.path.join(self.image_root, row["ImageID"])
        image = Image.open(image_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        text = row["Caption"]
        return image, str(text)

# Custom collate function to ensure text remains a list of strings
def custom_collate(batch):
    images, texts = zip(*batch)
    texts = [t.item() if isinstance(t, torch.Tensor) and t.ndim == 0 else str(t) for t in texts]
    return torch.stack(images), texts

# Prediction function
def predict_miniclip_ablation(
    model_path="best_miniclip_classifier.pt",
    csv_file="test_clean.csv",
    image_root="COMP5329S1A2Dataset/data",
    output_file="test_predictions.csv",
    proj_dim=1024,
    activation='relu',
    hidden_dims=[512, 256],
    use_dropout=True,
    dropout_rate=0.3,
    threshold=0.3,
    batch_size=32,
    use_batchnorm=False,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize model structure
    model = CLIPStyleModel(
        proj_dim=proj_dim,
        dropout_rate=dropout_rate,
        activation=activation,
        hidden_dims=hidden_dims,
        use_dropout=use_dropout,
        use_batchnorm=use_batchnorm
    )
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    # Load transforms and tokenizer
    transform= transforms.Compose([
        transforms.Resize(size=224, interpolation=transforms.InterpolationMode.BICUBIC),  # Resize image
        transforms.CenterCrop(224),                          # Center crop to 224x224
        transforms.RandomHorizontalFlip(),                   # Random horizontal flip
        transforms.ToTensor(),                               # Convert PIL image to tensor
        transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],  # Normalize (CLIP-style)
                             std=[0.26862954, 0.26130258, 0.27577711]),
    ])

    dataset = InferenceDataset(csv_file, image_root, transform)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=custom_collate)

    all_preds = []

    # Inference loop
    with torch.no_grad():
        for images, texts in tqdm(loader):
            images = images.to(device)

            # Pass raw text directly to model (let model tokenize internally)
            logits = model(images, texts)
            probs = torch.sigmoid(logits).cpu().numpy()
            preds = (probs > threshold).astype(int)
            all_preds.append(preds)

    if not all_preds:
        raise ValueError("No predictions were generated. Check dataset and model output.")

    all_preds = np.vstack(all_preds)

    # Prepare final output with label indices starting from 1 instead of 0
    final_output = []
    for idx, row in enumerate(all_preds):
        label_list = [str(i + 1) for i, v in enumerate(row) if v == 1]  # label indices start from 1
        image_id = f"{30000 + idx}.jpg"
        final_output.append({
            "ImageID": image_id,
            "Labels": " ".join(label_list)
        })

    # Save to CSV
    df_final = pd.DataFrame(final_output)
    df_final.to_csv(output_file, index=False)
    print(f"Saved {len(final_output)} predictions to {output_file}")