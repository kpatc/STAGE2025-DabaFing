"""
Module d'apprentissage par renforcement interactif (IRL)
avec fine-tuning supervisé basé sur les feedbacks humains
stockés dans un fichier CSV structuré
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import pandas as pd
from PIL import Image
from torchvision import transforms
from datetime import datetime
from tqdm import tqdm
import logging

logger = logging.getLogger(__name__)

class FeedbackDataset(Dataset):
    def __init__(self, dataframe, label_mapping, transform=None):
        self.df = dataframe
        self.transform = transform
        self.label_mapping = label_mapping

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image = Image.open(row['image_path']).convert("RGB")
        label_str = row['label']
        label = self.label_mapping[label_str]
        ridge = float(row['ridge_count'])
        if self.transform:
            image = self.transform(image)
        return image, (label, ridge)


class IRLFineTuner:
    def __init__(self, model, model_name="multihead_model.pth"):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = model.to(self.device)
        self.model_name = model_name

    def load_feedback_csv(self, csv_path):
        df = pd.read_csv(csv_path)
        if not {'image_path', 'label', 'ridge_count'}.issubset(df.columns):
            raise ValueError("Le CSV doit contenir les colonnes 'image_path', 'label', 'ridge_count'")
        return df

    def build_label_mapping(self, labels):
        unique_labels = sorted(labels.unique())
        return {label: idx for idx, label in enumerate(unique_labels)}

    def train_on_feedbacks(self, csv_path, epochs=3, lr=1e-4, batch_size=16):
        print("\n🔁 Starting IRL fine-tuning based on human feedback...")

        df = self.load_feedback_csv(csv_path)
        label_mapping = self.build_label_mapping(df['label'])

        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        dataset = FeedbackDataset(df, label_mapping, transform=transform)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        optimizer = optim.AdamW(self.model.parameters(), lr=lr, weight_decay=1e-2)
        cls_criterion = nn.CrossEntropyLoss()
        reg_criterion = nn.MSELoss()

        self.model.train()
        for epoch in range(epochs):
            running_cls_loss, running_reg_loss = 0.0, 0.0
            correct, total = 0, 0
            pbar = tqdm(loader, desc=f"IRL Epoch {epoch+1}/{epochs}")
            for images, (labels_cls, labels_ridge) in pbar:
                images = images.to(self.device)
                labels_cls = labels_cls.to(self.device)
                labels_ridge = labels_ridge.to(self.device)

                optimizer.zero_grad()
                out_cls, out_ridge = self.model(images)

                cls_loss = cls_criterion(out_cls, labels_cls)
                reg_loss = reg_criterion(out_ridge.squeeze(), labels_ridge.float())
                loss = cls_loss + 0.2 * reg_loss
                loss.backward()
                optimizer.step()

                running_cls_loss += cls_loss.item()
                running_reg_loss += reg_loss.item()
                preds = out_cls.argmax(dim=1)
                correct += (preds == labels_cls).sum().item()
                total += labels_cls.size(0)

                pbar.set_postfix({
                    "ClsLoss": f"{running_cls_loss / (pbar.n + 1):.4f}",
                    "RegLoss": f"{running_reg_loss / (pbar.n + 1):.4f}",
                    "Acc": f"{(100. * correct / total):.2f}%"
                })

        os.makedirs("irl_models", exist_ok=True)
        path = f"irl_models/{self.model_name.replace('.pth', '')}_IRL_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pth"
        torch.save(self.model.state_dict(), path)
        print(f"\n✅ IRL fine-tuned model saved to {path}")
        return path
