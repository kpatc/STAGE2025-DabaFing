import traceback
from flask import Flask, render_template, redirect, url_for, request, flash, jsonify
from flask_sqlalchemy import SQLAlchemy
from flask_bcrypt import Bcrypt
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from flask_cors import CORS
from flask_talisman import Talisman
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from flask_mail import Mail, Message
from itsdangerous import URLSafeTimedSerializer
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.models import load_model
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.optimizers import Adam
import cv2
import numpy as np
import os
from werkzeug.utils import secure_filename
from datetime import datetime
from itsdangerous import URLSafeTimedSerializer
from functools import wraps
from flask import abort
import torch
from model_utils import CLASS_NAMES, load_repvgg_model, preprocess_image, predict
from model_utils import load_repvgg_model, preprocess_image, predict
from model_utils import load_repvgg_model, preprocess_image, predict, CLASS_NAMES
import shutil
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import torch.nn as nn
from sklearn.model_selection import train_test_split
import pandas as pd
import json
from sqlalchemy.orm import joinedload
from flask import Response
import time
import json
import os
from PIL import Image
import torch
from torch.utils.data import Dataset
import torch
from torch.utils.data import DataLoader
import traceback
import os
from datetime import datetime
import json
import shutil
from flask import jsonify
from torchvision import transforms
import pandas as pd
from sklearn.model_selection import train_test_split
import torch.nn as nn
import torch.optim as optim
# Scheduler pour vérification automatique de l'entraînement
from flask import Flask, request
import atexit

# Initialisation de l'application Flask
app = Flask(__name__)

# ✅ Définir d'abord la clé secrète
app.config['SECRET_KEY'] = 'ma_clé_super_secrète_à_remplacer'

# ✅ Ensuite, tu peux utiliser app.config['SECRET_KEY']
secret_key = app.config['SECRET_KEY']
serializer = URLSafeTimedSerializer(secret_key)

# ✅ Fonctions liées aux tokens
def generate_reset_token(email):
    return serializer.dumps(email, salt='password-reset-salt')

def verify_reset_token(token, expiration=3600):
    try:
        email = serializer.loads(token, salt='password-reset-salt', max_age=expiration)
    except Exception:
        return None
    return email


# Configuration pour l'envoi d'email via Gmail
app.config['MAIL_SERVER'] = 'smtp.gmail.com'
app.config['MAIL_PORT'] = 587
app.config['MAIL_USE_TLS'] = True
app.config['MAIL_USE_SSL'] = False
app.config['MAIL_USERNAME'] = 'yourmail@gmail.com'
app.config['MAIL_PASSWORD'] = 'nkkc hsaa dztq wyma'
app.config['MAIL_DEFAULT_SENDER'] = 'yourmail@gmail.com'
app.config['MAIL_SUPPRESS_SEND'] = False
app.config['SECRET_KEY'] = os.getenv('SECRET_KEY', 'votre_cle_secrete')

# Configuration pour la classification d'empreintes
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg', 'bmp'}
app.config['MAX_FILE_SIZE'] = 8 * 1024 * 1024  # 8MB
app.config['REPVGG_MODEL_PATH'] = os.path.join('models', 'repvgg_best.pth')
app.config['CLASS_NAMES'] = CLASS_NAMES  # Utilisation des noms de classes définis dans model_utils.py

# Variable globale pour le scheduler
scheduler = None
# Chargement du modèle RepVGG
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
repvgg_model = None

try:
    repvgg_model = load_repvgg_model(app.config['REPVGG_MODEL_PATH'], device=device)
    print(f"✔ Modèle RepVGG chargé avec succès sur {device}")
except Exception as e:
    print(f"Erreur de chargement du modèle: {str(e)}")

# Création des dossiers si inexistants
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs('models', exist_ok=True)



# Initialisation des extensions
mail = Mail(app)
s = URLSafeTimedSerializer(app.config['SECRET_KEY'])
CORS(app, resources={
    r"/*": {
        "origins": "http://localhost:5173",
        "methods": ["GET", "POST", "PUT", "DELETE", "OPTIONS"],
        "allow_headers": ["Content-Type", "Authorization"],
        "supports_credentials": True,
        "expose_headers": ["Content-Type"]
    }
})

# Sécurité HTTPS et en-têtes
Talisman(app,
         content_security_policy={
             'default-src': [
                 '\'self\'',
                 'https://cdn.jsdelivr.net',
             ]
         },
         force_https=True)

# Limitation des requêtes
limiter = Limiter(
    get_remote_address,
    app=app,
    default_limits=["200 per day", "50 per hour"]
)

# Configuration de la base de données
app.config['SQLALCHEMY_DATABASE_URI'] = 'postgresql://postgres:user@localhost:5432/DabaFing'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

# Initialisation des extensions
db = SQLAlchemy(app)
bcrypt = Bcrypt(app)
login_manager = LoginManager(app)
login_manager.login_view = 'login'

# Modèle d'utilisateur
class Utilisateur(UserMixin, db.Model):
    __tablename__ = 'utilisateur'
    id_utilisateur = db.Column(db.Integer, primary_key=True)
    nom = db.Column(db.String(100), nullable=False)
    prenom = db.Column(db.String(100), nullable=False)
    email = db.Column(db.String(255), unique=True, nullable=False)
    mot_de_passe = db.Column(db.String(200), nullable=False)
    role = db.Column(db.String(20), nullable=False, default='utilisateur')

    def __repr__(self):
        return f"<Utilisateur {self.nom} {self.prenom}>"

    def get_id(self):
        return str(self.id_utilisateur)

@login_manager.user_loader
def load_user(user_id):
    return Utilisateur.query.get(int(user_id))

class ImageEmpreinte(db.Model):
    __tablename__ = 'imageempreinte'
    id_image = db.Column(db.Integer, primary_key=True)
    path_image = db.Column(db.Text, nullable=False)
    format = db.Column(db.String(50), nullable=False)
    empreinte = db.relationship('Empreinte', back_populates='image_empreinte', uselist=False)

class Empreinte(db.Model):
    __tablename__ = 'empreinte'
    id_empreinte = db.Column(db.Integer, primary_key=True)
    id_utilisateur = db.Column(db.Integer, db.ForeignKey('utilisateur.id_utilisateur'), nullable=False)
    id_image = db.Column(db.Integer, db.ForeignKey('imageempreinte.id_image'), nullable=False)
    type = db.Column(db.String(50), nullable=False)
    utilisateur = db.relationship('Utilisateur', backref='empreintes')
    image_empreinte = db.relationship('ImageEmpreinte', backref='empreintes')

class Analyse(db.Model):
    __tablename__ = 'Analyse'
    id_analyse = db.Column(db.Integer, primary_key=True)
    id_empreinte = db.Column(db.Integer, db.ForeignKey('empreinte.id_empreinte'), nullable=False)
    classification = db.Column(db.String(100), nullable=False)
    date_analyse = db.Column(db.DateTime, nullable=False, default=datetime.utcnow)
    nombre_de_cretes = db.Column(db.Integer, nullable=False, default=0)
    empreinte = db.relationship('Empreinte', backref='analyses')



# Fonctions utilitaires
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']





# Routes d'authentification
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        data = request.get_json() if request.is_json else request.form
        nom = data.get('nom')
        prenom = data.get('prenom')
        email = data.get('email')
        mot_de_passe = data.get('mot_de_passe')

        if not all([nom, prenom, email, mot_de_passe]):
            message = "Tous les champs sont requis."
            return jsonify({"message": message}) if request.is_json else flash(message, 'danger')

        if Utilisateur.query.filter_by(email=email).first():
            message = "Un utilisateur avec cet email existe déjà."
            return jsonify({"message": message}), 400 if request.is_json else flash(message, 'danger')

        mot_de_passe_hache = bcrypt.generate_password_hash(mot_de_passe).decode('utf-8')

        nouvel_utilisateur = Utilisateur(
            nom=nom,
            prenom=prenom,
            email=email,
            mot_de_passe=mot_de_passe_hache
        )
        db.session.add(nouvel_utilisateur)
        db.session.commit()

        message = "Compte créé avec succès !"
        if request.is_json:
            return jsonify({"message": message}), 200
        else:
            flash(message, 'success')
            return redirect(url_for('login'))

    return render_template('auth/register.html')


@app.route('/login', methods=['GET', 'POST'])
@limiter.limit("5 per minute", override_defaults=False)
def login():
    if request.method == 'POST':
        # Vérifie si les données sont en JSON
        if request.is_json:
            data = request.get_json()
            email = data.get('email')
            mot_de_passe = data.get('mot_de_passe')
        else:
            email = request.form.get('email')
            mot_de_passe = request.form.get('mot_de_passe')

        utilisateur = Utilisateur.query.filter_by(email=email).first()

        if utilisateur and bcrypt.check_password_hash(utilisateur.mot_de_passe, mot_de_passe):
            login_user(utilisateur)
            if request.is_json or request.headers.get('X-Requested-With') == 'XMLHttpRequest':
                return jsonify({
                    "message": "Connexion réussie",
                    "user": {
                        "id": utilisateur.id_utilisateur,
                        "email": utilisateur.email,
                        "role": utilisateur.role
                    }
                }), 200
            flash('Connexion réussie !', 'success')
            return redirect(url_for('profile'))
        else:
            if request.is_json or request.headers.get('X-Requested-With') == 'XMLHttpRequest':
                return jsonify({"message": "Email ou mot de passe incorrect"}), 401
            flash('Email ou mot de passe incorrect.', 'danger')
            return redirect(url_for('login'))

    return render_template('auth/login.html')

@app.route('/logout')
@login_required
def logout():
    logout_user()
    flash('Déconnexion réussie.', 'success')
    return redirect(url_for('home'))

@app.route('/profile')
@login_required
def profile():
    return render_template('profile.html', utilisateur=current_user)

# Routes pour la réinitialisation du mot de passe
# Route pour la réinitialisation du mot de passe (soumettre l'email)
@app.route('/forgot-password', methods=['GET', 'POST'])
def forgot_password():
    if request.method == 'POST':
        email = request.form.get('email')
        user = Utilisateur.query.filter_by(email=email).first()

        if user:
            # Génération du token de réinitialisation
            token = generate_reset_token(user.email)
            reset_url = url_for('reset_password', token=token, _external=True)

            # Envoi de l'email avec le lien
            msg = Message("Réinitialisation de votre mot de passe", recipients=[email])
            msg.body = f"Voici votre lien pour réinitialiser votre mot de passe : {reset_url}"

            # Envoi de l'email
            mail.send(msg)
            flash('Un lien de réinitialisation a été envoyé à votre email.', 'success')
            
            # Redirection vers la page de réinitialisation (avec le token dans l'URL)
            return render_template('message_confirmation.html')
  # Redirige vers la page de réinitialisation

        else:
            flash('Aucun utilisateur trouvé avec cet email.', 'danger')

    return render_template('auth/forgot_password.html')

# Route pour réinitialiser le mot de passe
@app.route('/reset-password/<token>', methods=['GET', 'POST'])
def reset_password(token):
    email = verify_reset_token(token)
    if not email:
        flash('Le lien est invalide ou a expiré.', 'danger')
        return redirect(url_for('home'))

    user = Utilisateur.query.filter_by(email=email).first()

    if request.method == 'POST':
        new_password = request.form.get('new_password')

        # Vérification que les mots de passe correspondent
        confirm_password = request.form.get('confirm_password')
        if new_password != confirm_password:
            flash('Les mots de passe ne correspondent pas.', 'danger')
            return redirect(url_for('reset_password', token=token))

        # Hachage et mise à jour du mot de passe
        hashed_password = bcrypt.generate_password_hash(new_password).decode('utf-8')
        user.mot_de_passe = hashed_password
        db.session.commit()

        flash('Votre mot de passe a été réinitialisé avec succès.', 'success')
        return redirect('http://localhost:5173/login')


    return render_template('auth/reset_password.html', token=token)

# Route pour l'analyse d'empreintes
@app.route('/fingerprint', methods=['POST'])
@login_required
def fingerprint_analysis():
    if not repvgg_model:
        return jsonify({'error': "Modèle non chargé"}), 503

    if 'file' not in request.files:
        return jsonify({'error': 'Aucun fichier fourni'}), 400
        
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'Aucun fichier sélectionné'}), 400
        
    if not (file and allowed_file(file.filename)):
        return jsonify({'error': 'Type de fichier non supporté'}), 400

    try:
        # Vérification taille fichier
        file.seek(0, os.SEEK_END)
        if file.tell() > app.config['MAX_FILE_SIZE']:
            return jsonify({'error': 'Fichier trop volumineux (>8MB)'}), 400
        file.seek(0)

        # Sauvegarde fichier
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        # Prétraitement et prédiction
        img_tensor = preprocess_image(filepath)
        if img_tensor is None:
            return jsonify({'error': "Erreur de traitement de l'image"}), 400
            
        prediction = predict(img_tensor, repvgg_model, device)
        if prediction is None:
            return jsonify({'error': "Erreur lors de la prédiction"}), 500

        # Enregistrement en base de données
        format_image = filename.rsplit('.', 1)[1].lower()
        nouvelle_image = ImageEmpreinte(path_image=filepath, format=format_image)
        db.session.add(nouvelle_image)
        db.session.commit()
        
        nouvelle_empreinte = Empreinte(
            id_utilisateur=current_user.id_utilisateur,
            id_image=nouvelle_image.id_image,
            type=prediction['class_name']
        )
        db.session.add(nouvelle_empreinte)
        db.session.commit()
        
        nouvelle_analyse = Analyse(
            id_empreinte=nouvelle_empreinte.id_empreinte,
            classification=prediction['class_name'],
            date_analyse=datetime.now(),
            nombre_de_cretes=prediction['ridge_count']
        )
        db.session.add(nouvelle_analyse)
        db.session.commit()

        return jsonify({
            'success': True,
            'filename': filename,
            'prediction': prediction['class_name'],
            'confidence': prediction['confidence'],
            'nombre_cretes': prediction['ridge_count'],
            'imageUrl': url_for('static', filename=f'uploads/{filename}', _external=True),
            'id_analyse': nouvelle_analyse.id_analyse
        })

    except Exception as e:
        db.session.rollback()
        return jsonify({'error': f"Erreur serveur: {str(e)}"}), 500

def is_admin():
    return current_user.is_authenticated and current_user.role == 'admin'

from functools import wraps
from flask import abort

def admin_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if not is_admin():
            abort(403)  # Accès interdit
        return f(*args, **kwargs)
    return decorated_function

@app.route('/admin/dashboard')
@login_required
@admin_required
def admin_dashboard():
    return render_template('admin/dashboard.html')

@app.route('/admin/users')
@login_required
@admin_required
def manage_users():
    users = Utilisateur.query.all()
    # Convert users to a list of dictionaries
    users_data = [{
        'id_utilisateur': user.id_utilisateur,
        'nom': user.nom,
        'prenom': user.prenom,
        'email': user.email,
        'role': user.role
    } for user in users]
    return jsonify(users_data)

@app.route('/admin/user/<int:user_id>/delete', methods=['POST'])
@login_required
@admin_required
def delete_user(user_id):
    user = Utilisateur.query.get_or_404(user_id)
    if user.id_utilisateur == current_user.id_utilisateur:
        flash("Vous ne pouvez pas supprimer votre propre compte ici", 'danger')
    else:
        db.session.delete(user)
        db.session.commit()
        flash("Utilisateur supprimé avec succès", 'success')
    return redirect(url_for('manage_users'))

@app.route('/admin/user/<int:user_id>/toggle-admin', methods=['POST'])
@login_required
@admin_required
def toggle_admin(user_id):
    user = Utilisateur.query.get_or_404(user_id)
    if user.id_utilisateur == current_user.id_utilisateur:
        flash("Vous ne pouvez pas modifier votre propre statut ici", 'danger')
    else:
        user.role = 'admin' if user.role != 'admin' else 'utilisateur'
        db.session.commit()
        flash(f"Statut admin {'activé' if user.role == 'admin' else 'désactivé'} pour {user.email}", 'success')
    return redirect(url_for('manage_users'))

def get_model_accuracy():
    accuracy_file = os.path.join('models', 'best_model_accuracy.txt')
    try:
        with open(accuracy_file, 'r') as f:
            for line in f:
                if 'Validation Accuracy:' in line:
                    # Extraire la valeur de précision (ex: "Validation Accuracy: 0.9878")
                    accuracy = line.split(':')[-1].strip()
                    return float(accuracy) * 100  # Convertir en pourcentage
            return None
    except Exception as e:
        print(f"Erreur lors de la lecture du fichier de précision: {str(e)}")
        return None
    
@app.route('/admin/model-accuracy')
@login_required
@admin_required
def model_accuracy():
    accuracy_file = os.path.join('models', 'best_model_accuracy.txt')
    try:
        with open(accuracy_file, 'r') as f:
            content = f.read()
            # Extraire accuracy et loss
            accuracy_line = [line for line in content.split('\n') if 'Validation Accuracy:' in line][0]
            loss_line = [line for line in content.split('\n') if 'Validation Loss:' in line][0]
            
            accuracy_value = float(accuracy_line.split(':')[-1].strip())
            loss_value = float(loss_line.split(':')[-1].strip())
            
            return jsonify({
                'accuracy': accuracy_value * 100,  # Convertir en pourcentage
                'loss_value': loss_value
            })
    except Exception as e:
        return jsonify({'error': "Impossible de récupérer les métriques du modèle"}), 500
    

# Modèle pour stocker les corrections
class CorrectionAnalyse(db.Model):
    __tablename__ = 'correction_analyse'
    id_correction = db.Column(db.Integer, primary_key=True)
    id_analyse = db.Column(db.Integer, db.ForeignKey('Analyse.id_analyse'), nullable=False)
    id_utilisateur = db.Column(db.Integer, db.ForeignKey('utilisateur.id_utilisateur'), nullable=False)
    classification_corrigee = db.Column(db.String(100), nullable=False)
    nombre_cretes_corrige = db.Column(db.Integer, nullable=False)
    date_correction = db.Column(db.DateTime, nullable=False, default=datetime.utcnow)
    statut = db.Column(db.String(20), nullable=False, default='en_attente')  # 'en_attente', 'validee', 'rejetee'
    commentaire = db.Column(db.Text)

    # Relation avec l'analyse originale
    analyse = db.relationship('Analyse', backref='corrections')
    utilisateur = db.relationship('Utilisateur', backref='corrections')

# Model pour stocker l'historique des entraînements incrémentaux
class EntrainementIncremental(db.Model):
    __tablename__ = 'entrainement_incremental'
    id = db.Column(db.Integer, primary_key=True)
    date_debut = db.Column(db.DateTime, nullable=False, default=datetime.utcnow)
    date_fin = db.Column(db.DateTime)
    nb_echantillons = db.Column(db.Integer, nullable=False)
    nb_epochs = db.Column(db.Integer, nullable=False)
    accuracy = db.Column(db.Float)
    loss = db.Column(db.Float)
    f1_score = db.Column(db.Float)
    modele_path = db.Column(db.String(255))
    backup_path = db.Column(db.String(255))
    distribution_classes = db.Column(db.Text)  # JSON stocké comme texte
    params_entrainement = db.Column(db.Text)   # JSON stocké comme texte
    status = db.Column(db.String(50), default='en_cours')  # en_cours, termine, erreur
    message_erreur = db.Column(db.Text)

# Nouvelle route pour soumettre une correction
@app.route('/fingerprint/correction', methods=['POST'])
@login_required
def submit_correction():
    data = request.get_json()
    
    # Validation des données
    if not data or 'id_analyse' not in data or 'classification' not in data or 'nombre_cretes' not in data:
        return jsonify({'error': 'Données de correction incomplètes'}), 400
    
    try:
        # Vérifier que l'analyse existe
        analyse = Analyse.query.get_or_404(data['id_analyse'])
        
        # Vérifier que l'utilisateur a le droit de corriger cette analyse
        empreinte = Empreinte.query.get(analyse.id_empreinte)
        if empreinte.id_utilisateur != current_user.id_utilisateur and current_user.role != 'admin':
            return jsonify({'error': 'Non autorisé à corriger cette analyse'}), 403
        
        # Vérifier que la classification proposée est valide
        if data['classification'] not in app.config['CLASS_NAMES']:
            return jsonify({'error': 'Classification invalide'}), 400
        
        # Vérifier que le nombre de crêtes est valide
        if not isinstance(data['nombre_cretes'], int) or data['nombre_cretes'] < 0:
            return jsonify({'error': 'Nombre de crêtes invalide'}), 400
        
        # Créer une nouvelle correction
        nouvelle_correction = CorrectionAnalyse(
            id_analyse=analyse.id_analyse,
            id_utilisateur=current_user.id_utilisateur,
            classification_corrigee=data['classification'],
            nombre_cretes_corrige=data['nombre_cretes'],
            commentaire=data.get('commentaire', '')
        )
        
        db.session.add(nouvelle_correction)
        db.session.commit()
        
        return jsonify({
            'success': True,
            'message': 'Correction soumise avec succès',
            'correction_id': nouvelle_correction.id_correction
        })
    
    except Exception as e:
        db.session.rollback()
        return jsonify({'error': f"Erreur lors de la soumission de la correction: {str(e)}"}), 500

# Route pour obtenir l'historique des corrections d'une analyse
@app.route('/fingerprint/<int:analyse_id>/corrections', methods=['GET', 'OPTIONS'])
@login_required
def get_corrections(analyse_id):
    if request.method == 'OPTIONS':
        return jsonify({'status': 'ok'}), 200
    
    try:
        analyse = Analyse.query.get_or_404(analyse_id)
        empreinte = Empreinte.query.get(analyse.id_empreinte)
        
        # Vérifier les permissions
        if empreinte.id_utilisateur != current_user.id_utilisateur and current_user.role != 'admin':
            return jsonify({'error': 'Non autorisé à voir ces corrections'}), 403
        
        corrections = CorrectionAnalyse.query.filter_by(id_analyse=analyse_id).all()
        
        return jsonify({
            'success': True,
            'corrections': [{
                'id': corr.id_correction,
                'user': corr.utilisateur.email,
                'date': corr.date_correction.isoformat(),
                'classification': corr.classification_corrigee,
                'nombre_cretes': corr.nombre_cretes_corrige,
                'statut': corr.statut,
                'commentaire': corr.commentaire
            } for corr in corrections]
        })
    
    except Exception as e:
        return jsonify({'error': f"Erreur lors de la récupération des corrections: {str(e)}"}), 500
    
# Route pour les admins pour valider/rejeter les corrections
@app.route('/admin/correction/<int:correction_id>/review', methods=['POST'])
@login_required
@admin_required
def review_correction(correction_id):
    data = request.get_json()
    
    if not data or 'action' not in data or data['action'] not in ['validate', 'reject']:
        return jsonify({'error': 'Action invalide'}), 400
    
    try:
        correction = CorrectionAnalyse.query.get_or_404(correction_id)
        analyse = correction.analyse
        
        if data['action'] == 'validate':
            # Mettre à jour l'analyse originale avec les valeurs corrigées
            analyse.classification = correction.classification_corrigee
            analyse.nombre_de_cretes = correction.nombre_cretes_corrige
            correction.statut = 'validee'
            
            db.session.commit()
            
            return jsonify({
                'success': True,
                'message': 'Correction validée et appliquée',
                'updated_analysis': {
                    'classification': analyse.classification,
                    'nombre_cretes': analyse.nombre_de_cretes
                }
            })
        
        else:  # 'reject'
            correction.statut = 'rejetee'
            db.session.commit()
            
            return jsonify({
                'success': True,
                'message': 'Correction rejetée'
            })
    
    except Exception as e:
        db.session.rollback()
        return jsonify({'error': f"Erreur lors du traitement de la correction: {str(e)}"}), 500
    

@app.route('/fingerprint/corrections/pending', methods=['GET'])
@login_required
@admin_required
def get_pending_corrections():
    try:
        pending_corrections = CorrectionAnalyse.query.filter_by(statut='en_attente').all()
        
        corrections_data = []
        for correction in pending_corrections:
            # Get the original analysis data
            original_analysis = correction.analyse
            original_empreinte = Empreinte.query.get(original_analysis.id_empreinte)
            image = ImageEmpreinte.query.get(original_empreinte.id_image)
            
            # Get the image URL
            image_url = url_for('static', filename=f'uploads/{os.path.basename(image.path_image)}', _external=True)
            
            corrections_data.append({
                'id': correction.id_correction,
                'id_analyse': original_analysis.id_analyse,
                'user_email': correction.utilisateur.email,
                'original_classification': original_analysis.classification,
                'classification_corrigee': correction.classification_corrigee,
                'original_ridges': original_analysis.nombre_de_cretes,
                'nombre_cretes_corrige': correction.nombre_cretes_corrige,
                'date_correction': correction.date_correction.isoformat(),
                'commentaire': correction.commentaire,
                'image_url': image_url  # Ajout de l'URL de l'image
            })
        
        return jsonify({
            'success': True,
            'corrections': corrections_data
        })
    
    except Exception as e:
        return jsonify({'error': f"Erreur lors de la récupération des corrections: {str(e)}"}), 500
    
@app.route('/user/analyses', methods=['GET', 'OPTIONS'])
@login_required
def get_user_analyses():
    if request.method == 'OPTIONS':
        response = jsonify({'status': 'ok'})
        response.headers.add('Access-Control-Allow-Origin', 'http://localhost:5173')
        response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
        response.headers.add('Access-Control-Allow-Methods', 'GET,OPTIONS')
        response.headers.add('Access-Control-Allow-Credentials', 'true')
        return response
    
    try:
        # Récupère toutes les analyses de l'utilisateur connecté avec les empreintes et images associées
        # Version optimisée avec jointures explicites
        query = db.session.query(
            Analyse,
            Empreinte,
            ImageEmpreinte
        ).select_from(Analyse)\
         .join(
             Empreinte, 
             Analyse.id_empreinte == Empreinte.id_empreinte
         ).join(
             ImageEmpreinte,
             Empreinte.id_image == ImageEmpreinte.id_image
         ).filter(
             Empreinte.id_utilisateur == current_user.id_utilisateur
         ).order_by(
             Analyse.date_analyse.desc()
         )

        # Exécution de la requête
        results = query.all()

        # Formatage des données
        analyses_data = []
        for analyse, empreinte, image in results:
            # Récupère les corrections associées avec une requête séparée
            corrections = db.session.query(CorrectionAnalyse)\
                .filter(
                    CorrectionAnalyse.id_analyse == analyse.id_analyse
                ).order_by(
                    CorrectionAnalyse.date_correction.desc()
                ).all()

            corrections_data = [{
                'id': c.id_correction,
                'date': c.date_correction.isoformat(),
                'classification': c.classification_corrigee,
                'nombre_cretes': c.nombre_cretes_corrige,
                'statut': c.statut,
                'commentaire': c.commentaire
            } for c in corrections]

            # Construction du chemin de l'image
            image_filename = os.path.basename(image.path_image)
            image_url = url_for(
                'static', 
                filename=f'uploads/{image_filename}', 
                _external=True
            )

            analyses_data.append({
                'id_analyse': analyse.id_analyse,
                'date': analyse.date_analyse.isoformat(),
                'image_url': image_url,
                'classification': analyse.classification,
                'nombre_cretes': analyse.nombre_de_cretes,
                'corrections': corrections_data
            })

        return jsonify({
            'success': True,
            'analyses': analyses_data
        })

    except Exception as e:
        app.logger.error(f"Erreur dans get_user_analyses: {str(e)}")
        app.logger.error(traceback.format_exc())
        return jsonify({
            'success': False,
            'error': f"Erreur lors de la récupération de l'historique: {str(e)}"
        }), 500
        
@app.route('/user/analyses/<int:analysis_id>', methods=['GET'])
@login_required
def get_single_analysis(analysis_id):
    try:
        # Vérifie que l'analyse appartient bien à l'utilisateur
        analyse = db.session.query(Analyse)\
            .join(Empreinte, Analyse.id_empreinte == Empreinte.id_empreinte)\
            .filter(
                Analyse.id_analyse == analysis_id,
                Empreinte.id_utilisateur == current_user.id_utilisateur
            )\
            .first_or_404()

        # Récupère les données associées
        empreinte = Empreinte.query.get(analyse.id_empreinte)
        image = ImageEmpreinte.query.get(empreinte.id_image)
        
        # Récupère les corrections
        corrections = CorrectionAnalyse.query\
            .filter_by(id_analyse=analysis_id)\
            .order_by(CorrectionAnalyse.date_correction.desc())\
            .all()

        corrections_data = [{
            'id': c.id_correction,
            'date': c.date_correction.isoformat(),
            'classification': c.classification_corrigee,
            'nombre_cretes': c.nombre_cretes_corrige,
            'statut': c.statut,
            'commentaire': c.commentaire
        } for c in corrections]

        return jsonify({
            'success': True,
            'analysis': {
                'id': analyse.id_analyse,
                'date': analyse.date_analyse.isoformat(),
                'image_url': url_for('static', filename=f'uploads/{os.path.basename(image.path_image)}', _external=True),
                'classification': analyse.classification,
                'nombre_cretes': analyse.nombre_de_cretes,
                'corrections': corrections_data
            }
        })

    except Exception as e:
        return jsonify({
            'success': False,
            'error': f"Erreur lors de la récupération de l'analyse: {str(e)}"
        }), 404 if "404" in str(e) else 500
        

@app.route('/fingerprint/multiple', methods=['POST'])
@login_required
def multiple_fingerprint_analysis():
    if not repvgg_model:
        return jsonify({'error': "Modèle non chargé"}), 503

    if 'files' not in request.files:
        return jsonify({'error': 'Aucun fichier fourni'}), 400

    # Récupérer tous les fichiers
    files = request.files.getlist('files')
    
    if len(files) != 10:
        return jsonify({'error': 'Veuillez uploader exactement 10 images (5 doigts par main)'}), 400

    try:
        results = []
        finger_names = [
            "Pouce droit", "Index droit", "Majeur droit", "Annulaire droit", "Auriculaire droit",
            "Pouce gauche", "Index gauche", "Majeur gauche", "Annulaire gauche", "Auriculaire gauche"
        ]

        for i, file in enumerate(files):
            if file.filename == '':
                return jsonify({'error': f'Aucun fichier sélectionné pour {finger_names[i]}'}), 400

            if not (file and allowed_file(file.filename)):
                return jsonify({'error': f'Type de fichier non supporté pour {finger_names[i]}'}), 400

            # Vérification taille fichier
            file.seek(0, os.SEEK_END)
            if file.tell() > app.config['MAX_FILE_SIZE']:
                return jsonify({'error': f'Fichier trop volumineux (>8MB) pour {finger_names[i]}'}), 400
            file.seek(0)

            # Sauvegarde fichier
            filename = secure_filename(f"{finger_names[i]}_{file.filename}")
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            # Prétraitement et prédiction
            img_tensor = preprocess_image(filepath)
            if img_tensor is None:
                return jsonify({'error': f"Erreur de traitement de l'image pour {finger_names[i]}"}), 400

            prediction = predict(img_tensor, repvgg_model, device)
            if prediction is None:
                return jsonify({'error': f"Erreur lors de la prédiction pour {finger_names[i]}"}), 500

            # Enregistrement en base de données
            format_image = filename.rsplit('.', 1)[1].lower()
            nouvelle_image = ImageEmpreinte(path_image=filepath, format=format_image)
            db.session.add(nouvelle_image)
            db.session.commit()
            
            nouvelle_empreinte = Empreinte(
                id_utilisateur=current_user.id_utilisateur,
                id_image=nouvelle_image.id_image,
                type=f"{finger_names[i]}: {prediction['class_name']}"
            )
            db.session.add(nouvelle_empreinte)
            db.session.commit()
            
            nouvelle_analyse = Analyse(
                id_empreinte=nouvelle_empreinte.id_empreinte,
                classification=prediction['class_name'],
                date_analyse=datetime.now(),
                nombre_de_cretes=prediction['ridge_count']
            )
            db.session.add(nouvelle_analyse)
            db.session.commit()

            results.append({
                'finger_name': finger_names[i],
                'filename': filename,
                'prediction': prediction['class_name'],
                'confidence': prediction['confidence'],
                'nombre_cretes': prediction['ridge_count'],
                'imageUrl': url_for('static', filename=f'uploads/{filename}', _external=True),
                'id_analyse': nouvelle_analyse.id_analyse
            })

        return jsonify({
            'success': True,
            'results': results
        })

    except Exception as e:
        db.session.rollback()
        return jsonify({'error': f"Erreur serveur: {str(e)}"}), 500
    

def prepare_training_data(min_samples=2):
    """
    Prépare les données de réentraînement avec validation renforcée des classes
    
    Args:
        min_samples (int): Nombre minimum d'échantillons requis par classe (défaut: 2)
    
    Returns:
        str: Chemin vers le fichier CSV des données préparées
        
    Raises:
        ValueError: Si les données ne satisfont pas les exigences de validation
    """
    try:
        # Récupération des données avec jointures
        corrections = db.session.query(
            CorrectionAnalyse,
            Analyse,
            Empreinte,
            ImageEmpreinte
        ).join(
            Analyse, CorrectionAnalyse.id_analyse == Analyse.id_analyse
        ).join(
            Empreinte, Analyse.id_empreinte == Empreinte.id_empreinte
        ).join(
            ImageEmpreinte, Empreinte.id_image == ImageEmpreinte.id_image
        ).filter(
            CorrectionAnalyse.statut == 'validee'
        ).all()
        
        if not corrections:
            raise ValueError("Aucune correction validée disponible pour l'entraînement")
        
        # Création du DataFrame et analyse des classes
        data = []
        class_counts = {}
        
        for correction, analyse, empreinte, image in corrections:
            class_name = correction.classification_corrigee
            data.append({
                'image_path': image.path_image,
                'label': class_name,
                'ridge_count': correction.nombre_cretes_corrige,
                'correction_id': correction.id_correction,
                'date_validation': correction.date_correction.strftime('%Y-%m-%d')
            })
            class_counts[class_name] = class_counts.get(class_name, 0) + 1
        
        # Validation avancée des données
        problematic_classes = {cls: count for cls, count in class_counts.items() if count < min_samples}
        sufficient_classes = {cls: count for cls, count in class_counts.items() if count >= min_samples}
        
        if problematic_classes:
            # Création du rapport détaillé
            report = {
                "metadata": {
                    "generated_at": datetime.now().isoformat(),
                    "min_samples_required": min_samples,
                    "total_samples": sum(class_counts.values()),
                    "total_classes": len(class_counts)
                },
                "class_distribution": {
                    "problematic": problematic_classes,
                    "sufficient": sufficient_classes
                },
                "analysis": {
                    "min_samples": min(class_counts.values()),
                    "max_samples": max(class_counts.values()),
                    "avg_samples": sum(class_counts.values()) / len(class_counts),
                    "recommendation": {
                        "minimum_to_add": {cls: min_samples - count for cls, count in problematic_classes.items()},
                        "priority_classes": sorted(problematic_classes.keys(), key=lambda x: problematic_classes[x])
                    }
                },
                "raw_data_preview": data[:3]  # Exemple des premières entrées
            }
            
            # Sauvegarde du rapport
            os.makedirs('training_reports', exist_ok=True)
            report_path = os.path.join('training_reports', 
                                     f"data_validation_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
            
            with open(report_path, 'w', encoding='utf-8') as f:
                json.dump(report, f, indent=2, ensure_ascii=False)
            
            # Préparation du message d'erreur clair
            error_msg = (
                f"Données insuffisantes pour l'entraînement:\n"
                f"- Classes problématiques ({len(problematic_classes)}): {', '.join(problematic_classes.keys())}\n"
                f"- Échantillons par classe: Min {min(class_counts.values())}, Max {max(class_counts.values())}, Moy {sum(class_counts.values())/len(class_counts):.1f}\n"
                f"- Rapport complet sauvegardé: {report_path}"
            )
            
            raise ValueError(error_msg)
        
        # Création du DataFrame final avec vérification des chemins d'images
        df = pd.DataFrame(data)
        
        # Vérification que les fichiers image existent
        missing_images = []
        for img_path in df['image_path']:
            if not os.path.exists(img_path):
                missing_images.append(img_path)
        
        if missing_images:
            raise ValueError(
                f"{len(missing_images)} images manquantes sur {len(df)}. "
                f"Premiers fichiers manquants: {missing_images[:5]}"
            )
        
        # Sauvegarde des données
        os.makedirs('training_data', exist_ok=True)
        csv_path = os.path.join('training_data', 
                              f"training_data_{datetime.now().strftime('%Y%m%d')}.csv")
        
        df.to_csv(csv_path, index=False)
        
        # Sauvegarde des métadonnées
        metadata = {
            "generated_at": datetime.now().isoformat(),
            "total_samples": len(df),
            "classes_count": len(class_counts),
            "class_distribution": class_counts,
            "data_path": csv_path
        }
        
        metadata_path = os.path.join('training_data', 'metadata.json')
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2)
        
        return csv_path

    except Exception as e:
        app.logger.error(f"Erreur dans prepare_training_data: {str(e)}")
        app.logger.error(traceback.format_exc())
        raise

# Function pour verifier les conditions d'entraînement incrémental
def check_incremental_training_conditions():
    """
    Vérifie si les conditions sont réunies pour un entraînement incrémental
    """
    try:
        # Vérifier s'il y a suffisamment d'images
        correction_count = db.session.query(CorrectionAnalyse).filter_by(statut='validee').count()
        if correction_count < 200:
            return False, f"Nombre de corrections insuffisant: {correction_count}/200"
        
        # Vérifier la représentation des classes
        corrections = db.session.query(CorrectionAnalyse).filter_by(statut='validee').all()
        
        # Compter les occurrences de chaque classe
        class_counts = {}
        for correction in corrections:
            class_name = correction.classification_corrigee
            class_counts[class_name] = class_counts.get(class_name, 0) + 1
        
        # Vérifier si toutes les classes sont présentes
        missing_classes = set(CLASS_NAMES) - set(class_counts.keys())
        if missing_classes:
            return False, f"Classes manquantes: {', '.join(missing_classes)}"
        
        # Vérifier qu'il n'y a pas d'entraînement en cours
        training_in_progress = db.session.query(EntrainementIncremental).filter_by(status='en_cours').first()
        if training_in_progress:
            return False, "Un entraînement est déjà en cours"
        
        # Tout est OK
        return True, class_counts
        
    except Exception as e:
        app.logger.error(f"Erreur lors de la vérification des conditions d'entraînement: {str(e)}")
        return False, f"Erreur: {str(e)}"
    

# Importez d'abord l'entraîneur incrémental
from irl_system.incremental_trainer import IncrementalTrainer

# Fonction d'entraînement incrémental
def perform_incremental_training(app_instance):
    """
    Effectue l'entraînement incrémental si les conditions sont remplies
    """
    try:
        # Vérifier les conditions
        conditions_met, result = check_incremental_training_conditions()
        if not conditions_met:
            app.logger.info(f"Conditions d'entraînement non remplies: {result}")
            return None
        
        # Créer une entrée dans la base de données
        training_record = EntrainementIncremental(
            nb_echantillons=sum(result.values()),
            nb_epochs=3,  # valeur par défaut
            status='en_cours',
            distribution_classes=json.dumps(result)
        )
        db.session.add(training_record)
        db.session.commit()
        
        try:
            # Préparation des données
            csv_path = prepare_training_data(min_samples=2)
            
            # Initialisation de l'entraîneur
            trainer = IncrementalTrainer(app_instance)
            
            # Configuration et lancement de l'entraînement
            training_results = trainer.train(
                csv_path=csv_path,
                epochs=3,
                learning_rate=0.0001,
                freeze_backbone=True,
                batch_size=8,
                load_pretrained=True,
                fine_tuning_strategy='progressive'  # Utilise la stratégie progressive
            )
            
            if not training_results.get('success', False):
                raise Exception(training_results.get('error', 'Erreur inconnue'))
            
            # Créer une sauvegarde du modèle actuel
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_path = os.path.join('models', f'repvgg_backup_{timestamp}.pth')
            shutil.copy(app_instance.config['REPVGG_MODEL_PATH'], backup_path)
            
            # Déployer le nouveau modèle
            shutil.copy(training_results['model_path'], app_instance.config['REPVGG_MODEL_PATH'])
            
            # Mettre à jour les métriques
            final_metrics = training_results.get('final_metrics', {})
            
            # Mise à jour du record d'entraînement
            training_record.date_fin = datetime.utcnow()
            training_record.accuracy = final_metrics.get('accuracy', 0) * 100  # Convertir en pourcentage
            training_record.f1_score = final_metrics.get('f1_macro', 0)
            training_record.modele_path = training_results['model_path']
            training_record.backup_path = backup_path
            training_record.params_entrainement = json.dumps({
                'epochs': training_results.get('training_history', [{}])[-1].get('epoch', 0),
                'learning_rate': training_results.get('training_history', [{}])[-1].get('learning_rate', 0),
                'fine_tuning_strategy': 'progressive'
            })
            training_record.status = 'termine'
            
            db.session.commit()
            
            # Recharger le modèle
            global repvgg_model
            repvgg_model = load_repvgg_model(app_instance.config['REPVGG_MODEL_PATH'], device=device)
            
            return training_record
            
        except Exception as e:
            # En cas d'erreur, mettre à jour le record
            training_record.status = 'erreur'
            training_record.message_erreur = str(e)
            training_record.date_fin = datetime.utcnow()
            db.session.commit()
            
            app.logger.error(f"Erreur lors de l'entraînement incrémental: {str(e)}")
            app.logger.error(traceback.format_exc())
            return None
            
    except Exception as e:
        app.logger.error(f"Erreur globale de l'entraînement incrémental: {str(e)}")
        app.logger.error(traceback.format_exc())
        return None

# Route pour vérifier les conditions d'entraînement incrémental
@app.route('/admin/incremental-training/check', methods=['GET'])
@login_required
@admin_required
def check_incremental_training():
    """
    Vérifie si les conditions sont réunies pour l'entraînement incrémental
    """
    try:
        conditions_met, result = check_incremental_training_conditions()
        
        if conditions_met:
            return jsonify({
                'ready': True,
                'class_distribution': result,
                'total_samples': sum(result.values())
            })
        else:
            return jsonify({
                'ready': False,
                'message': result
            })
            
    except Exception as e:
        return jsonify({
            'ready': False,
            'error': str(e)
        }), 500


# Route pour démarrer l'entraînement incrémental manuellement
@app.route('/admin/incremental-training/start', methods=['POST'])
@login_required
@admin_required
def start_incremental_training():
    """
    Lance l'entraînement incrémental manuellement
    """
    try:
        # Vérifier les conditions
        conditions_met, result = check_incremental_training_conditions()
        if not conditions_met:
            return jsonify({
                'success': False,
                'message': f"Conditions d'entraînement non remplies: {result}"
            }), 400
            
        # Lancer l'entraînement de façon asynchrone
        import threading
        thread = threading.Thread(target=perform_incremental_training, args=(app,))
        thread.daemon = True
        thread.start()
        
        return jsonify({
            'success': True,
            'message': "Entraînement incrémental démarré"
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

# Route pour obtenir l'historique des entraînements incrémentaux
@app.route('/admin/incremental-training/history', methods=['GET'])
@login_required
@admin_required
def get_incremental_training_history():
    """
    Récupère l'historique des entraînements incrémentiels
    """
    try:
        page = request.args.get('page', 1, type=int)
        per_page = request.args.get('per_page', 10, type=int)
        
        # Récupérer l'historique paginé
        training_history = EntrainementIncremental.query\
            .order_by(EntrainementIncremental.date_debut.desc())\
            .paginate(page=page, per_page=per_page, error_out=False)
        
        # Formater les résultats
        history_data = []
        for record in training_history.items:
            history_data.append({
                'id': record.id,
                'date_debut': record.date_debut.isoformat(),
                'date_fin': record.date_fin.isoformat() if record.date_fin else None,
                'duree': (record.date_fin - record.date_debut).total_seconds() if record.date_fin else None,
                'nb_echantillons': record.nb_echantillons,
                'nb_epochs': record.nb_epochs,
                'accuracy': record.accuracy,
                'f1_score': record.f1_score,
                'status': record.status,
                'classes': json.loads(record.distribution_classes) if record.distribution_classes else {}
            })
        
        return jsonify({
            'success': True,
            'history': history_data,
            'pagination': {
                'page': training_history.page,
                'per_page': training_history.per_page,
                'total_pages': training_history.pages,
                'total_items': training_history.total
            }
        })
        
    except Exception as e:
        app.logger.error(f"Erreur lors de la récupération de l'historique: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

# Route pour obtenir les détails d'un entraînement incrémental spécifique
@app.route('/admin/incremental-training/<int:training_id>', methods=['GET'])
@login_required
@admin_required
def get_training_details(training_id):
    """
    Récupère les détails d'un entraînement incrémental spécifique
    """
    try:
        record = EntrainementIncremental.query.get_or_404(training_id)
        
        # Formater les résultats détaillés
        details = {
            'id': record.id,
            'date_debut': record.date_debut.isoformat(),
            'date_fin': record.date_fin.isoformat() if record.date_fin else None,
            'duree': str(record.date_fin - record.date_debut) if record.date_fin else None,
            'nb_echantillons': record.nb_echantillons,
            'nb_epochs': record.nb_epochs,
            'accuracy': record.accuracy,
            'f1_score': record.f1_score,
            'modele_path': record.modele_path,
            'backup_path': record.backup_path,
            'status': record.status,
            'message_erreur': record.message_erreur,
            'distribution_classes': json.loads(record.distribution_classes) if record.distribution_classes else {},
            'params_entrainement': json.loads(record.params_entrainement) if record.params_entrainement else {}
        }
        
        return jsonify({
            'success': True,
            'details': details
        })
        
    except Exception as e:
        app.logger.error(f"Erreur lors de la récupération des détails: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/admin/check-training-data', methods=['GET'])
@login_required
@admin_required
def check_training_data():
    try:
        min_samples = request.args.get('min_samples', default=2, type=int)
        
        # Récupération des corrections validées avec leur classification
        corrections = CorrectionAnalyse.query.filter_by(statut='validee').all()
        
        if not corrections:
            return jsonify({
                'ready': False,
                'message': 'Aucune correction validée disponible',
                'total_samples': 0,
                'class_distribution': {}
            })

        # Calcul de la distribution des classes
        class_counts = {}
        for correction in corrections:
            class_name = correction.classification_corrigee
            class_counts[class_name] = class_counts.get(class_name, 0) + 1
        
        # Identification des classes problématiques
        problematic_classes = {
            cls: count for cls, count in class_counts.items() 
            if count < min_samples
        }
        
        return jsonify({
            'ready': len(problematic_classes) == 0,
            'class_distribution': class_counts,
            'problematic_classes': problematic_classes,
            'min_samples_required': min_samples,
            'total_samples': len(corrections),
            'statistics': {
                'min': min(class_counts.values()) if class_counts else 0,
                'max': max(class_counts.values()) if class_counts else 0,
                'avg': sum(class_counts.values())/len(class_counts) if class_counts else 0
            }
        })

    except Exception as e:
        app.logger.error(f"Erreur dans check_training_data: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e),
            'traceback': traceback.format_exc()
        }), 500

        
@app.route('/admin/model-status', methods=['GET'])
@login_required
@admin_required
def model_status():
    """Endpoint pour vérifier l'état et les métriques du modèle"""
    try:
        # Vérifie si des corrections validées sont disponibles
        corrections_count = CorrectionAnalyse.query.filter_by(statut='validee').count()
        
        # Récupère les métriques du modèle original
        original_metrics = {}
        accuracy_file = os.path.join('models', 'best_model_accuracy.txt')
        if os.path.exists(accuracy_file):
            with open(accuracy_file, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if 'Validation Accuracy:' in line:
                        # Gestion du pourcentage dans la valeur
                        acc_str = line.split(':')[-1].strip().replace('%', '')
                        original_metrics['accuracy'] = float(acc_str)
                    elif 'Validation Loss:' in line:
                        loss_str = line.split(':')[-1].strip()
                        original_metrics['loss'] = float(loss_str)
        
        # Récupère les métriques du dernier réentraînement
        retrained_metrics = {}
        retrained_file = os.path.join('models', 'retrained_model_metrics.txt')
        if os.path.exists(retrained_file):
            with open(retrained_file, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if 'Validation Accuracy:' in line:
                        acc_str = line.split(':')[-1].strip().replace('%', '')
                        retrained_metrics['accuracy'] = float(acc_str)
                    elif 'Validation Loss:' in line:
                        loss_str = line.split(':')[-1].strip()
                        retrained_metrics['loss'] = float(loss_str)
        
        return jsonify({
            'success': True,
            'original_model_metrics': original_metrics,
            'last_retrained_metrics': retrained_metrics if retrained_metrics else None,
            'available_corrections': corrections_count,
            'model_path': app.config['REPVGG_MODEL_PATH'],
            'device': str(device)
        })
    
    except Exception as e:
        app.logger.error(f"Erreur dans model_status: {str(e)}")
        return jsonify({
            'success': False,
            'error': "Erreur lors de la récupération des métriques du modèle",
            'details': str(e)
        }), 500
        

#Service de vérification automatique des conditions d'entraînement
def setup_automatic_training_check():
    """
    Configure la vérification automatique des conditions d'entraînement
    """
    from apscheduler.schedulers.background import BackgroundScheduler
    
    def auto_check_and_train():
        with app.app_context():
            app.logger.info("Vérification automatique des conditions d'entraînement...")
            conditions_met, result = check_incremental_training_conditions()
            if conditions_met:
                app.logger.info("Conditions d'entraînement remplies, lancement de l'entraînement...")
                perform_incremental_training(app)
            else:
                app.logger.info(f"Conditions d'entraînement non remplies: {result}")
    
    # Créer le scheduler
    scheduler = BackgroundScheduler()
    scheduler.add_job(
        auto_check_and_train, 
        'interval', 
        hours=24,  # Vérifier une fois par jour
        id='auto_train_check',
        replace_existing=True
    )
    
    # Démarrer le scheduler
    scheduler.start()
    app.logger.info("Vérification automatique de l'entraînement configurée")
    return scheduler

# Démarrer la vérification automatique si l'application est lancée directement
if __name__ == "__main__":
    # Configurer le scheduler avant de lancer l'app
    scheduler = setup_automatic_training_check()
    atexit.register(lambda: scheduler.shutdown() if scheduler else None)

    
    try:
        app.run(host='0.0.0.0', port=5000, debug=True)
    except (KeyboardInterrupt, SystemExit):
        # S'assurer que le scheduler est arrêté proprement
        if scheduler:
            scheduler.shutdown()

