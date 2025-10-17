import torch
import torch.nn as nn
import timm
import torch.nn.functional as F
from PIL import Image
import numpy as np
import cv2
from torchvision import transforms

# Définir les noms des classes exacts comme dans votre modèle
CLASS_NAMES = [
    'Accidental',
    'Ambiguous',
    'Tented Arch',
    'Simple Arch', 
    'Tented Arch with Loop',
    'Ulnar/Radial Loop',
    'Spiral Whorl',
    'Target Centric Whorl',
    'Ulnar/Radial Peacock Eye Whorl',
    'Double Loop',
    'Elongated Spiral Whorl', 
    'Elongated Target Centric Whorl'
]

class RepVGGMultiTask(nn.Module):
    def __init__(self, num_classes=12, dropout_rate=0.5):
        super().__init__()
        self.base_model = timm.create_model('repvgg_a2', pretrained=False)
        in_features = self.base_model.num_features
        self.base_model.head = nn.Identity()
        self.dropout = nn.Dropout(p=dropout_rate)
        self.class_head = nn.Linear(in_features, num_classes)
        self.ridge_head = nn.Linear(in_features, 1)

    def forward(self, x):
        features = self.base_model.forward_features(x)
        features = F.adaptive_avg_pool2d(features, (1, 1))
        features = torch.flatten(features, 1)
        features = self.dropout(features)
        class_logits = self.class_head(features)
        ridge_output = self.ridge_head(features)
        return class_logits, ridge_output.squeeze(1)

def load_repvgg_model(model_path, device='cpu'):
    model = RepVGGMultiTask(num_classes=len(CLASS_NAMES), dropout_rate=0.5)
    
    checkpoint = torch.load(model_path, map_location=device)
    
    # Gérer DataParallel si utilisé pendant l'entraînement
    if isinstance(checkpoint, torch.nn.DataParallel):
        checkpoint = checkpoint.module.state_dict()
    
    new_state_dict = {}
    for k, v in checkpoint.items():
        name = k[7:] if k.startswith('module.') else k
        new_state_dict[name] = v
    
    model.load_state_dict(new_state_dict)
    model.to(device)
    model.eval()
    return model

def get_transforms():
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

def preprocess_image(image_path, target_size=(224, 224)):
    try:
        img = Image.open(image_path).convert('RGB')
        transform = get_transforms()
        img = img.resize(target_size)
        img_tensor = transform(img)
        return img_tensor.unsqueeze(0)
    except Exception as e:
        print(f"Erreur de prétraitement: {str(e)}")
        return None

def predict(image_tensor, model, device='cpu'):
    try:
        with torch.no_grad():
            image_tensor = image_tensor.to(device)
            class_logits, ridge_count = model(image_tensor)
            
            class_probs = F.softmax(class_logits, dim=1)
            confidence, predicted_class = torch.max(class_probs, 1)
            
            return {
                'class_idx': predicted_class.item(),
                'class_name': CLASS_NAMES[predicted_class.item()],
                'confidence': confidence.item(),
                'ridge_count': int(round(ridge_count.item()))
            }
    except Exception as e:
        print(f"Erreur de prédiction: {str(e)}")
        return None