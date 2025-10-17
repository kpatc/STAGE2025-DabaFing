import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
from pathlib import Path
from typing import Dict, Tuple, List
import logging
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score, classification_report
import timm
from collections import OrderedDict
import os
from datetime import datetime
from pathlib import Path

class CorrectionDataset(Dataset):
    """Dataset pour les corrections d'IRL - Compatible avec RepVGG du notebook"""
    
    def __init__(self, csv_path: str, transform=None):
        self.data = pd.read_csv(csv_path)
        self.transform = transform
        
        # ✅ MAPPING DE CONVERSION: Noms complets → Abréviations
        self.fullname_to_abbrev = {
            'Accidental': 'AC',
            'Ambiguous': 'Ambiguous',
            'Tented Arch': 'TA',
            'Simple Arch': 'SA',
            'Ulnar/Radial Peacock Eye Whorl': 'UPE_RPE',
            'Spiral Whorl': 'SW',
            'Target Centric Whorl': 'TCW',
            'Double Loop': 'DL',
            'Elongated Spiral Whorl': 'ESW',
            'Elongated Target Centric Whorl': 'ECW',
            'TAUL_TARL': 'TAUL_TARL',
            'UL_RL': 'UL_RL'
        }
        
        # ✅ EXACT MAPPING du notebook RepVGG
        self.label_to_idx = {
            'AC': 0, 'Ambiguous': 1, 'TA': 2, 'SA': 3, 'TAUL_TARL': 4, 'UL_RL': 5,
            'SW': 6, 'TCW': 7, 'UPE_RPE': 8, 'DL': 9, 'ESW': 10, 'ECW': 11
        }
        
        # Mapping inverse
        self.idx_to_label = {v: k for k, v in self.label_to_idx.items()}
        self.class_names = [self.idx_to_label[i] for i in sorted(self.idx_to_label.keys())]
        
        # Classes sans crêtes (du notebook)
        self.zero_ridge_classes = {'AC', 'Ambiguous', 'TA', 'SA'}
        
        # Valider les données
        self._validate_data()
        
    def _validate_data(self):
        """Valider et nettoyer les données avec conversion automatique"""
        print(f"🔍 Validation des données IRL...")
        
        # Vérifier les colonnes requises
        required_cols = ['image_path', 'label', 'ridge_count']
        missing_cols = [col for col in required_cols if col not in self.data.columns]
        if missing_cols:
            raise ValueError(f"Colonnes manquantes dans le CSV: {missing_cols}")
        
        # ✅ CONVERSION AUTOMATIQUE: Noms complets → Abréviations
        print("🔄 Conversion des noms de classes...")
        
        original_labels = self.data['label'].unique()
        converted_count = 0
        unknown_labels = []
        
        for i, row in self.data.iterrows():
            original_label = row['label']
            
            # Essayer la conversion
            if original_label in self.fullname_to_abbrev:
                converted_label = self.fullname_to_abbrev[original_label]
                self.data.at[i, 'label'] = converted_label
                converted_count += 1
            elif original_label in self.label_to_idx:
                # Déjà une abréviation valide
                pass
            else:
                unknown_labels.append(original_label)
        
        print(f"✅ {converted_count} labels convertis")
        
        # Afficher les conversions effectuées
        unique_conversions = {}
        for orig, abbrev in self.fullname_to_abbrev.items():
            if orig in original_labels:
                unique_conversions[orig] = abbrev
        
        if unique_conversions:
            print("📝 Conversions effectuées:")
            for orig, abbrev in unique_conversions.items():
                count = sum(1 for label in original_labels if label == orig)
                print(f"  '{orig}' → '{abbrev}' ({count} échantillons)")
        
        # Gérer les labels inconnus
        if unknown_labels:
            print(f"⚠️ Labels inconnus (seront ignorés): {set(unknown_labels)}")
            self.data = self.data[~self.data['label'].isin(unknown_labels)]
        
        # Vérifier que tous les labels sont maintenant valides
        remaining_labels = set(self.data['label'].unique())
        invalid_labels = remaining_labels - set(self.label_to_idx.keys())
        if invalid_labels:
            print(f"❌ Labels toujours invalides: {invalid_labels}")
            self.data = self.data[self.data['label'].isin(self.label_to_idx.keys())]
        
        # Vérifier les fichiers d'images
        valid_indices = []
        for idx, row in self.data.iterrows():
            image_path = row['image_path']
            if Path(image_path).exists():
                valid_indices.append(idx)
            else:
                print(f"⚠️ Image manquante: {image_path}")
        
        self.data = self.data.iloc[valid_indices].reset_index(drop=True)
        
        if len(self.data) == 0:
            raise ValueError("Aucune image valide trouvée après validation")
        
        # Statistiques finales
        print(f"✅ Dataset validé: {len(self.data)} échantillons")
        print(f"📊 Distribution des corrections (après conversion):")
        for label, count in self.data['label'].value_counts().items():
            print(f"  {label}: {count} corrections")
        
        # ✅ CRÉER MAPPING DYNAMIQUE basé sur les données présentes
        self.present_labels = sorted(self.data['label'].unique())
        self.present_label_to_idx = {label: idx for idx, label in enumerate(self.present_labels)}
        self.present_idx_to_label = {idx: label for label, idx in self.present_label_to_idx.items()}
        
        print(f"📋 Classes présentes dans les données: {self.present_labels}")
        print(f"🔄 Mapping dynamique créé: {self.present_label_to_idx}")
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        
        # Charger l'image en grayscale (comme dans le notebook)
        image_path = row['image_path']
        image = Image.open(image_path).convert('L')  # Grayscale
        
        if self.transform:
            image = self.transform(image)
        
        # Labels (maintenant déjà convertis en abréviations)
        label_str = row['label']
        
        # ✅ UTILISER LE MAPPING DYNAMIQUE
        label_idx = self.present_label_to_idx[label_str]
        ridge_count = float(row['ridge_count'])
        
        return image, label_idx, ridge_count

# ✅ CLASSE RepVGG EXACTE du notebook
class RepVGGMultiTask(nn.Module):
    def __init__(self, num_classes=12, dropout_rate=0.5):
        super().__init__()
        # Load pre-trained RepVGG-A2 from timm
        self.base_model = timm.create_model('repvgg_a2', pretrained=True)

        in_features = self.base_model.num_features

        # Replace the original classifier head with an Identity layer.
        self.base_model.head = nn.Identity()

        # Add a Dropout layer
        self.dropout = nn.Dropout(p=dropout_rate)

        # Classification head
        self.class_head = nn.Linear(in_features, num_classes)

        # Ridge count regression head
        self.ridge_head = nn.Linear(in_features, 1)

    def forward(self, x):
        features = self.base_model.forward_features(x)

        features = F.adaptive_avg_pool2d(features, (1, 1))
        features = torch.flatten(features, 1)
        features = self.dropout(features)

        class_logits = self.class_head(features)
        ridge_output = self.ridge_head(features)
        return class_logits, ridge_output.squeeze(1)

class IncrementalTrainer:
    """Entraîneur incrémental pour RepVGG Multi-Task - Compatible avec le notebook"""
    
    def __init__(self, app):
        self.app = app
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.logger = logging.getLogger(__name__)
        
        # ✅ EXACT TRANSFORM du notebook RepVGG
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.Grayscale(num_output_channels=3),  # Convert to 3 channels
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])  # Normalize grayscale
        ])
        
        print(f"🏗️ Trainer RepVGG Multi-Task initialisé")
        print(f"📱 Device: {self.device}")

    def train(self, csv_path: str, epochs: int = 10, learning_rate: float = 0.0001, 
          freeze_backbone: bool = True, batch_size: int = 8, 
          load_pretrained: bool = True) -> Dict:
        """
        Entraînement incrémental avec RepVGG Multi-Task
        """
        try:
            self.logger.info(f"🚀 Début entraînement incrémental RepVGG")
            
            # Nettoyage GPU
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                print(f"🧹 GPU memory cleared")
            
            # ✅ NOUVEAU: Stocker les mappings pour les tests
            self.fullname_to_abbrev = {
                'Accidental': 'AC',
                'Ambiguous': 'Ambiguous',
                'Tented Arch': 'TA',
                'Simple Arch': 'SA',
                'Ulnar/Radial Peacock Eye Whorl': 'UPE_RPE',
                'Spiral Whorl': 'SW',
                'Target Centric Whorl': 'TCW',
                'Double Loop': 'DL',
                'Elongated Spiral Whorl': 'ESW',
                'Elongated Target Centric Whorl': 'ECW',
                'TAUL_TARL': 'TAUL_TARL',
                'UL_RL': 'UL_RL'
            }
            
            self.label_to_idx = {
                'AC': 0, 'Ambiguous': 1, 'TA': 2, 'SA': 3, 'TAUL_TARL': 4, 'UL_RL': 5,
                'SW': 6, 'TCW': 7, 'UPE_RPE': 8, 'DL': 9, 'ESW': 10, 'ECW': 11
            }
            
            self.idx_to_label = {v: k for k, v in self.label_to_idx.items()}
            
            # 1. Préparation des données AVANT le modèle
            self.logger.info("📊 Préparation des données de corrections...")
            train_loader, val_loader, data_stats = self._prepare_correction_data(csv_path, batch_size)
            
            if data_stats['train_samples'] == 0:
                return {'success': False, 'error': 'Aucune donnée d\'entraînement disponible'}
            
            # 2. Création/Chargement du modèle RepVGG
            num_classes = data_stats['num_classes_present']
            print(f"🏗️ Configuration du modèle RepVGG avec {num_classes} classes")
            
            if load_pretrained:
                model = self._load_pretrained_repvgg_model(num_classes)
            else:
                model = RepVGGMultiTask(num_classes=num_classes, dropout_rate=0.5)
                model = model.to(self.device)
            
            # Vérifier le modèle
            total_params = sum(p.numel() for p in model.parameters())
            print(f"✅ Modèle RepVGG configuré avec {total_params:,} paramètres")
            
            # Test rapide
            with torch.no_grad():
                dummy_input = torch.randn(1, 3, 224, 224).to(self.device)
                class_out, ridge_out = model(dummy_input)
                print(f"🔍 Sortie modèle: Classification {class_out.shape}, Ridge {ridge_out.shape}")
            
            # 3. Configuration du fine-tuning
            if freeze_backbone:
                self._freeze_backbone_intelligently(model)
            
            # 4. Optimiseur avec LR plus élevé pour les nouvelles têtes
            trainable_params = [p for p in model.parameters() if p.requires_grad]
            optimizer = torch.optim.Adam(trainable_params, lr=learning_rate * 10, weight_decay=1e-5)  # LR x10
            print(f"🔥 Learning rate augmenté à {learning_rate * 10} pour les nouvelles têtes")
            
            # 5. Critères de perte (identiques au notebook)
            criterion_cls = nn.CrossEntropyLoss()
            criterion_reg = nn.MSELoss()
            
            # 6. Entraînement
            self.logger.info("🔄 Début de l'entraînement...")
            training_history = self._training_loop(
                model, train_loader, val_loader, optimizer, 
                criterion_cls, criterion_reg, epochs, data_stats
            )
            
            # 7. Évaluation finale
            final_metrics = self._evaluate_final_model(model, val_loader, criterion_cls, criterion_reg, data_stats)
            
            # 8. Sauvegarde
            model_path = self._save_model(model, training_history, final_metrics, data_stats)
            
            return {
                'success': True,
                'model_path': model_path,
                'training_history': training_history,
                'final_metrics': final_metrics,
                'data_stats': data_stats,
                'message': f'Fine-tuning RepVGG terminé avec {final_metrics["accuracy"]:.2%} d\'accuracy'
            }
            
        except Exception as e:
            self.logger.error(f"Erreur entraînement incrémental: {str(e)}")
            import traceback
            traceback.print_exc()
            return {'success': False, 'error': str(e)} 
    
    def _load_pretrained_repvgg_model(self, num_classes):
        """Charger le modèle RepVGG pré-entraîné et conserver la connaissance des têtes"""
        try:
            # Utiliser model_utils pour charger le modèle pré-entraîné
            from model_utils import load_repvgg_model
            
            model_path = self.app.config.get('REPVGG_MODEL_PATH')
            if not model_path or not Path(model_path).exists():
                print(f"⚠️ Modèle pré-entraîné non trouvé: {model_path}")
                print("🔄 Création d'un nouveau modèle RepVGG...")
                return RepVGGMultiTask(num_classes=num_classes, dropout_rate=0.5).to(self.device)
            
            # Charger le modèle pré-entraîné
            print(f"📥 Chargement du modèle RepVGG pré-entraîné: {model_path}")
            pretrained_model = load_repvgg_model(model_path, device=self.device)
            
            # Obtenir le nombre de classes du modèle pré-entraîné
            pretrained_num_classes = pretrained_model.class_head.out_features
            print(f"📊 Modèle pré-entraîné: {pretrained_num_classes} classes")
            print(f"📊 Nouveau modèle requis: {num_classes} classes")
            
            # ✅ NOUVELLE STRATÉGIE: Conserver les têtes quand c'est possible
            if num_classes == pretrained_num_classes:
                # Cas idéal: même nombre de classes, on garde tout!
                print(f"✅ Même nombre de classes: conservation complète des têtes!")
                return pretrained_model
            
            # Créer un nouveau modèle avec le bon nombre de classes
            new_model = RepVGGMultiTask(num_classes=num_classes, dropout_rate=0.5)
            
            # Copier les poids
            pretrained_state = pretrained_model.state_dict()
            new_state = new_model.state_dict()
            
            # Copier tous les poids compatibles (backbone ET têtes partielles)
            transferred_count = 0
            for key in pretrained_state.keys():
                # Si c'est la dernière couche de classification, on ne peut pas la copier
                # si la dimension ne correspond pas
                if key == 'class_head.weight' or key == 'class_head.bias':
                    if pretrained_state[key].shape != new_state[key].shape:
                        continue
                
                # Pour tous les autres poids, copier si les dimensions correspondent
                if key in new_state and pretrained_state[key].shape == new_state[key].shape:
                    new_state[key] = pretrained_state[key]
                    transferred_count += 1
                    
                    # Indiquer quelles parties sont transférées
                    if 'class_head' in key or 'ridge_head' in key:
                        print(f"  ✅ Transféré (tête): {key}")
            
            # Charger les poids transférés
            new_model.load_state_dict(new_state)
            new_model = new_model.to(self.device)
            
            print(f"✅ Transfert réussi: {transferred_count} couches copiées")
            
            # ✨ NOUVEAU: Initialisation spéciale des têtes non transférées
            if num_classes > pretrained_num_classes:
                print(f"🔄 Initialisation spéciale des nouvelles classes...")
                # Initialiser les nouvelles classes avec la moyenne des poids existants
                with torch.no_grad():
                    if hasattr(new_model.class_head, 'weight'):
                        existing_weights = new_model.class_head.weight[:pretrained_num_classes].mean(dim=0)
                        for i in range(pretrained_num_classes, num_classes):
                            new_model.class_head.weight[i] = existing_weights + torch.randn_like(existing_weights) * 0.02
            
            return new_model
            
        except Exception as e:
            self.logger.error(f"Erreur chargement modèle pré-entraîné: {e}")
            print(f"⚠️ Erreur chargement, création d'un nouveau modèle: {e}")
            return RepVGGMultiTask(num_classes=num_classes, dropout_rate=0.5).to(self.device)

    def _test_pretrained_model(self, pretrained_model):
        """Tester le modèle pré-entraîné sur quelques échantillons"""
        try:
            pretrained_model.eval()
            
            # Créer un petit dataset de test avec les vraies classes (0-11)
            test_samples = []
            test_labels = []
            
            for idx, row in self.test_data.iterrows():
                if len(test_samples) >= 10:  # Limiter à 10 échantillons
                    break
                    
                image_path = row['image_path']
                if Path(image_path).exists():
                    image = Image.open(image_path).convert('L')
                    image = self.transform(image).unsqueeze(0).to(self.device)
                    
                    # Utiliser le mapping original (0-11) pour le modèle pré-entraîné
                    original_label = row['label']
                    if original_label in self.fullname_to_abbrev:
                        original_label = self.fullname_to_abbrev[original_label]
                    
                    if original_label in self.label_to_idx:
                        original_idx = self.label_to_idx[original_label]
                        test_samples.append(image)
                        test_labels.append(original_idx)
            
            if not test_samples:
                return 0.0
            
            # Tester le modèle pré-entraîné
            correct = 0
            total = len(test_samples)
            
            with torch.no_grad():
                for image, true_label in zip(test_samples, test_labels):
                    class_logits, _ = pretrained_model(image)
                    _, predicted = torch.max(class_logits, 1)
                    
                    if predicted.item() == true_label:
                        correct += 1
                        
                    print(f"  📝 Vrai: {self.idx_to_label[true_label]}, Prédit: {self.idx_to_label[predicted.item()]}")
            
            accuracy = correct / total
            return accuracy
            
        except Exception as e:
            print(f"⚠️ Erreur test modèle pré-entraîné: {e}")
            return 0.0

    def _test_new_model_heads(self, new_model):
        """Tester le modèle avec nouvelles têtes sur les données de correction"""
        try:
            new_model.eval()
            
            # Utiliser les vraies données de correction
            test_samples = []
            test_labels = []
            
            # Prendre quelques échantillons de validation
            for idx, row in self.correction_data.iterrows():
                if len(test_samples) >= 10:
                    break
                    
                image_path = row['image_path']
                if Path(image_path).exists():
                    image = Image.open(image_path).convert('L')
                    image = self.transform(image).unsqueeze(0).to(self.device)
                    
                    # Utiliser le mapping dynamique (0-8)
                    label_str = row['label']
                    if label_str in self.correction_dataset.present_label_to_idx:
                        label_idx = self.correction_dataset.present_label_to_idx[label_str]
                        test_samples.append(image)
                        test_labels.append(label_idx)
            
            if not test_samples:
                return 0.0
            
            # Tester le nouveau modèle
            correct = 0
            total = len(test_samples)
            
            with torch.no_grad():
                for image, true_label in zip(test_samples, test_labels):
                    class_logits, _ = new_model(image)
                    _, predicted = torch.max(class_logits, 1)
                    
                    if predicted.item() == true_label:
                        correct += 1
                        
                    true_label_name = self.correction_dataset.present_idx_to_label[true_label]
                    pred_label_name = self.correction_dataset.present_idx_to_label[predicted.item()]
                    print(f"  📝 Vrai: {true_label_name}, Prédit: {pred_label_name}")
            
            accuracy = correct / total
            return accuracy
            
        except Exception as e:
            print(f"⚠️ Erreur test nouvelles têtes: {e}")
            return 0.0
        
    
    def _prepare_correction_data(self, csv_path: str, batch_size: int) -> Tuple[DataLoader, DataLoader, Dict]:
        """Préparer les données de corrections"""
        
        # Créer le dataset de corrections
        dataset = CorrectionDataset(csv_path, transform=self.transform)
        
        # ✅ NOUVEAU: Stocker les données pour les tests
        self.correction_dataset = dataset
        self.correction_data = dataset.data
        
        # Créer aussi un dataset de test (pour tester le modèle pré-entraîné)
        # Utiliser quelques échantillons des données de correction
        self.test_data = dataset.data.copy()
        
        # Statistiques basées sur les données présentes
        present_labels = dataset.present_labels
        present_label_counts = dataset.data['label'].value_counts()
        
        print(f"📊 Distribution des corrections:")
        for label, count in present_label_counts.items():
            print(f"  {label}: {count} corrections")
        
        # Division train/val (80/20)
        train_size = max(1, int(0.8 * len(dataset)))
        val_size = len(dataset) - train_size
        
        # Seed fixe pour reproductibilité
        generator = torch.Generator().manual_seed(42)
        train_dataset, val_dataset = torch.utils.data.random_split(
            dataset, [train_size, val_size], generator=generator
        )
        
        # DataLoaders
        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True, num_workers=0
        )
        val_loader = DataLoader(
            val_dataset, batch_size=batch_size, shuffle=False, num_workers=0
        )
        
        data_stats = {
            'total_samples': len(dataset),
            'train_samples': train_size,
            'val_samples': val_size,
            'num_classes_present': len(present_labels),
            'present_labels': present_labels,
            'present_label_to_idx': dataset.present_label_to_idx,
            'present_idx_to_label': dataset.present_idx_to_label,
            'class_distribution': present_label_counts.to_dict()
        }
        
        print(f"📈 Données préparées: {train_size} train, {val_size} val")
        print(f"📋 Classes présentes: {present_labels}")
        
        return train_loader, val_loader, data_stats

    
    def _freeze_backbone_intelligently(self, model):
        """Gel intelligent du backbone pour fine-tuning"""
        print("🔍 Configuration du fine-tuning RepVGG...")
        
        frozen_params = 0
        trainable_params = 0
        
        for name, param in model.named_parameters():
            # Stratégie: garder seulement les têtes entraînables
            if 'class_head' in name or 'ridge_head' in name:
                param.requires_grad = True
                trainable_params += param.numel()
                print(f"🔥 Entraînable: {name}")
            else:
                param.requires_grad = False
                frozen_params += param.numel()
        
        print(f"📊 Fine-tuning configuré:")
        print(f"❄️  Paramètres gelés: {frozen_params:,}")
        print(f"🔥 Paramètres entraînables: {trainable_params:,}")
        
        if trainable_params == 0:
            raise ValueError("Aucun paramètre entraînable configuré!")
        
        return frozen_params, trainable_params
    
    def _training_loop(self, model, train_loader, val_loader, optimizer, 
                      criterion_cls, criterion_reg, epochs, data_stats):
        """Boucle d'entraînement optimisée (identique au notebook)"""
        
        training_history = []
        best_val_accuracy = 0.0
        REG_IMPORTANCE = 0.5  # Même valeur que le notebook
        
        for epoch in range(epochs):
            print(f"\n🔄 Epoch {epoch+1}/{epochs}")
            
            # =========================
            # PHASE D'ENTRAÎNEMENT
            # =========================
            model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0
            
            for batch_idx, (images, class_labels, ridge_counts) in enumerate(train_loader):
                images = images.to(self.device)
                class_labels = class_labels.to(self.device)
                ridge_counts = ridge_counts.to(self.device).float()
                
                optimizer.zero_grad()
                
                # Forward pass
                class_logits, ridge_preds = model(images)
                
                # ✅ CALCUL DE PERTE identique au notebook
                loss_cls = criterion_cls(class_logits, class_labels)
                loss_reg = criterion_reg(ridge_preds, ridge_counts)
                loss = (1 - REG_IMPORTANCE) * loss_cls + REG_IMPORTANCE * loss_reg
                
                # Backward pass
                loss.backward()
                
                # Gradient clipping pour stabilité
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                optimizer.step()
                
                # Métriques
                train_loss += loss.item()
                _, predicted = torch.max(class_logits, 1)
                train_total += class_labels.size(0)
                train_correct += (predicted == class_labels).sum().item()
                
                if batch_idx % 2 == 0:
                    print(f"  Batch {batch_idx+1}/{len(train_loader)} - Loss: {loss.item():.4f}")
            
            # =========================
            # PHASE DE VALIDATION
            # =========================
            val_results = self._validate_epoch(model, val_loader, criterion_cls, criterion_reg, REG_IMPORTANCE)
            
            # Historique
            epoch_metrics = {
                'epoch': epoch + 1,
                'train_loss': train_loss / len(train_loader),
                'train_accuracy': train_correct / train_total,
                'val_loss': val_results['loss'],
                'val_accuracy': val_results['accuracy'],
                'val_f1': val_results['f1'],
                'learning_rate': optimizer.param_groups[0]['lr']
            }
            training_history.append(epoch_metrics)
            
            # Affichage
            print(f"📊 Epoch {epoch+1} Results:")
            print(f"  Train - Loss: {epoch_metrics['train_loss']:.4f}, Acc: {epoch_metrics['train_accuracy']:.2%}")
            print(f"  Val   - Loss: {epoch_metrics['val_loss']:.4f}, Acc: {epoch_metrics['val_accuracy']:.2%}, F1: {epoch_metrics['val_f1']:.4f}")
            print(f"  LR: {epoch_metrics['learning_rate']:.6f}")
            
            # Tracking du meilleur modèle
            if val_results['accuracy'] > best_val_accuracy:
                best_val_accuracy = val_results['accuracy']
                print(f"🎯 Nouveau meilleur modèle! Accuracy: {best_val_accuracy:.2%}")
        
        return training_history
    
    def _validate_epoch(self, model, val_loader, criterion_cls, criterion_reg, reg_importance):
        """Validation d'une époque"""
        model.eval()
        val_loss = 0.0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for images, class_labels, ridge_counts in val_loader:
                images = images.to(self.device)
                class_labels = class_labels.to(self.device)
                ridge_counts = ridge_counts.to(self.device).float()
                
                class_logits, ridge_preds = model(images)
                
                # Perte
                loss_cls = criterion_cls(class_logits, class_labels)
                loss_reg = criterion_reg(ridge_preds, ridge_counts)
                loss = (1 - reg_importance) * loss_cls + reg_importance * loss_reg
                
                val_loss += loss.item()
                
                # Prédictions
                _, predicted = torch.max(class_logits, 1)
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(class_labels.cpu().numpy())
        
        # Métriques
        accuracy = accuracy_score(all_labels, all_preds)
        f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
        
        return {
            'loss': val_loss / len(val_loader),
            'accuracy': accuracy,
            'f1': f1
        }
    
    def _evaluate_final_model(self, model, val_loader, criterion_cls, criterion_reg, data_stats):
        """Évaluation finale détaillée"""
        model.eval()
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for images, class_labels, ridge_counts in val_loader:
                images = images.to(self.device)
                class_labels = class_labels.to(self.device)
                
                class_logits, ridge_preds = model(images)
                
                _, predicted = torch.max(class_logits, 1)
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(class_labels.cpu().numpy())
        
        # Métriques finales
        accuracy = accuracy_score(all_labels, all_preds)
        f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
        
        # ✅ CORRECTION: Utiliser les labels présents uniquement
        present_labels = data_stats['present_labels']
        unique_labels = sorted(set(all_labels))
        
        # Classification report avec les bonnes classes
        report = classification_report(
            all_labels, all_preds, 
            labels=unique_labels,
            target_names=[present_labels[i] for i in unique_labels],
            output_dict=True,
            zero_division=0
        )
        
        return {
            'accuracy': accuracy,
            'f1_macro': f1,
            'classification_report': report,
            'predictions': all_preds,
            'true_labels': all_labels,
            'present_labels': present_labels
        }
    
    def _save_model(self, model, training_history, final_metrics, data_stats):
        """Sauvegarde le modèle fine-tuné"""
        try:
            # Créer le dossier de sauvegarde
            save_dir = Path('models')
            save_dir.mkdir(exist_ok=True)
            
            # Nom du fichier avec timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            model_filename = f"repvgg_irl_finetuned_{timestamp}.pth"
            model_path = save_dir / model_filename
            
            # Sauvegarder le modèle
            torch.save({
                'model_state_dict': model.state_dict(),
                'training_history': training_history,
                'final_metrics': final_metrics,
                'data_stats': data_stats,
                'timestamp': timestamp,
                'model_type': 'RepVGG_IRL_FineTuned'
            }, model_path)
            
            # ✨ NOUVEAU: Sauvegarder les métriques au format requis
            self._save_metrics_for_api(final_metrics, training_history)
            
            print(f"💾 Modèle RepVGG fine-tuné sauvegardé: {model_path}")
            return str(model_path)
            
        except Exception as e:
            self.logger.error(f"Erreur sauvegarde modèle: {e}")
            raise

    def _save_metrics_for_api(self, final_metrics, training_history):
        """Sauvegarde les métriques au format spécifique pour l'API"""
        try:
            # Chemin du fichier de métriques
            save_dir = Path('models')
            metrics_file =save_dir / 'retrained_model_metrics.txt'
            
            # Récupérer l'accuracy depuis les métriques finales
            accuracy = final_metrics.get('accuracy', 0.0)
            
            # Pour la loss, essayer plusieurs sources possibles
            loss = None
            
            # 1. D'abord chercher dans les métriques finales
            if 'loss' in final_metrics:
                loss = final_metrics['loss']
            
            # 2. Sinon, chercher dans le dernier epoch de l'historique
            elif training_history and len(training_history) > 0:
                if 'val_loss' in training_history[-1]:
                    loss = training_history[-1]['val_loss']
                elif 'train_loss' in training_history[-1]:
                    loss = training_history[-1]['train_loss']
            
            # 3. Si toujours pas trouvé, utiliser une valeur par défaut
            if loss is None:
                loss = 0.0
            
            # Écrire les métriques au format exact requis
            with open(metrics_file, 'w', encoding='utf-8') as f:
                f.write(f"Validation Accuracy: {accuracy:.4f}\n")
                f.write(f"Validation Loss: {loss:.4f}\n")
            
            print(f"📊 Métriques sauvegardées dans {metrics_file}")
            
        except Exception as e:
            self.logger.error(f"Erreur sauvegarde métriques API: {e}")
            print(f"⚠️ Attention: Impossible de sauvegarder les métriques pour l'API: {e}")
            # Ne pas faire échouer tout le processus pour cette erreur