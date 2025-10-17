# test_incremental_trainer.py
import os
import sys
sys.path.append('.')

from irl_system.incremental_trainer import IncrementalTrainer

# Configuration minimale pour test
class MockApp:
    def __init__(self):
        self.config = {
            'REPVGG_MODEL_PATH': 'models/repvgg_best.pth'  # Chemin vers ton modèle
        }

# Test
if __name__ == "__main__":
    app = MockApp()
    trainer = IncrementalTrainer(app)

    csv_path='training_data/training_data_20250629.csv'
    print("🔧 Démarrage du test d'entraînement incrémental")
    # Lancer l'entraînement
    results = trainer.train(
        csv_path=csv_path,
        epochs=3,
        learning_rate=0.00001,
        freeze_backbone=True,
        batch_size=8,
        load_pretrained=True
    )
    
    print("Résultats:", results)