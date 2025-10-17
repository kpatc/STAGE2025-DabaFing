#!/usr/bin/env python3
from flask import Flask
from flask_bcrypt import Bcrypt
import sys

# Créer une mini-application Flask pour utiliser bcrypt de manière identique à l'app
app = Flask(__name__)
bcrypt = Bcrypt(app)

def hash_password(password):
    """Génère un hash de mot de passe compatible avec l'application"""
    hashed = bcrypt.generate_password_hash(password).decode('utf-8')
    return hashed

if __name__ == "__main__":
    if len(sys.argv) > 1:
        # Récupérer le mot de passe depuis les arguments
        password = sys.argv[1]
    else:
        # Demander le mot de passe en mode interactif
        password = input("Entrez le mot de passe à hasher: ")
    
    # Générer et afficher le hash
    hashed_password = hash_password(password)
    print("\n=== RÉSULTAT ===")
    print(f"Mot de passe original: {password}")
    print(f"Hash généré: {hashed_password}")
    print("\n=== REQUÊTE SQL ===")
    print(f"INSERT INTO utilisateur (nom, prenom, email, mot_de_passe, role)")
    print(f"VALUES ('Admin', 'Super', 'admin@dabafing.com', '{hashed_password}', 'admin');")