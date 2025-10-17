CREATE TABLE  Utilisateur(
    id_utilisateur SERIAL PRIMARY KEY,
    nom VARCHAR(100) NOT NULL,
    prenom VARCHAR(100) NOT NULL,
    email VARCHAR(255) UNIQUE NOT NULL ,
    mot_de_passe TEXT NOT NULL,
    role VARCHAR(20) CHECK (role IN ('admin', 'utilisateur')) NOT NULL
);

CREATE TABLE ImageEmpreinte (
    id_image SERIAL PRIMARY KEY,
    path_image TEXT NOT NULL,
    format VARCHAR(50) NOT NULL
);


CREATE TABLE Empreinte (
    id_empreinte SERIAL PRIMARY KEY,
    id_utilisateur INT NOT NULL,
    id_image INT NOT NULL,
    type VARCHAR(50) NOT NULL,
    FOREIGN KEY (id_utilisateur) REFERENCES Utilisateur(id_utilisateur) ON DELETE CASCADE,
    FOREIGN KEY (id_image) REFERENCES ImageEmpreinte(id_image) ON DELETE CASCADE
);

CREATE TABLE "Analyse" (
    id_analyse SERIAL PRIMARY KEY,
    id_empreinte INT NOT NULL,
    date_analyse TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    nombre_de_cretes INT NOT NULL,
    classification VARCHAR(100) NOT NULL,
    FOREIGN KEY (id_empreinte) REFERENCES Empreinte(id_empreinte) ON DELETE CASCADE
);

CREATE TABLE ModeleIA (
    id_modele SERIAL PRIMARY KEY,
    accuracy_actuelle DECIMAL(5,2) NOT NULL,
    accuracy_cible DECIMAL(5,2) NOT NULL
);

CREATE TABLE Feedback (
    id_feedback SERIAL PRIMARY KEY,
    id_analyse INT NOT NULL,
    id_utilisateur INT NOT NULL,
    commentaire TEXT NOT NULL,
    validation BOOLEAN DEFAULT FALSE,
    FOREIGN KEY (id_analyse) REFERENCES "Analyse"(id_analyse) ON DELETE CASCADE,
    FOREIGN KEY (id_utilisateur) REFERENCES Utilisateur(id_utilisateur) ON DELETE CASCADE
);

CREATE TABLE ExportResultat (
    id_export SERIAL PRIMARY KEY,
    id_analyse INT NOT NULL,
    format_export VARCHAR(50) NOT NULL,
    path_fich TEXT NOT NULL,
    FOREIGN KEY (id_analyse) REFERENCES "Analyse"(id_analyse) ON DELETE CASCADE
);

CREATE TABLE Rapport (
    id_rapport SERIAL PRIMARY KEY,
    id_analyse INT NOT NULL,
    format VARCHAR(50) NOT NULL,
    FOREIGN KEY (id_analyse) REFERENCES "Analyse"(id_analyse) ON DELETE CASCADE
);

CREATE TABLE correction_analyse (
    id_correction SERIAL PRIMARY KEY,
    id_analyse INTEGER NOT NULL,
    id_utilisateur INTEGER NOT NULL,
    classification_corrigee VARCHAR(100) NOT NULL,
    nombre_cretes_corrige INTEGER NOT NULL,
    date_correction TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    statut VARCHAR(20) NOT NULL DEFAULT 'en_attente',
    commentaire TEXT,
    FOREIGN KEY (id_analyse) REFERENCES Analyse(id_analyse),
    FOREIGN KEY (id_utilisateur) REFERENCES utilisateur(id_utilisateur)
);

CREATE TABLE entrainement_incremental (
    id SERIAL PRIMARY KEY,
    date_debut TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    date_fin TIMESTAMP,
    nb_echantillons INTEGER NOT NULL,
    nb_epochs INTEGER NOT NULL,
    accuracy FLOAT,
    loss FLOAT,
    f1_score FLOAT,
    modele_path VARCHAR(255),
    backup_path VARCHAR(255),
    distribution_classes TEXT,  -- JSON stocké comme texte
    params_entrainement TEXT,   -- JSON stocké comme texte
    status VARCHAR(50) DEFAULT 'en_cours',
    message_erreur TEXT
);

-- Ajouter un index pour améliorer les performances des requêtes par statut
CREATE INDEX idx_entrainement_status ON entrainement_incremental(status);

-- Ajouter un index pour les requêtes par date
CREATE INDEX idx_entrainement_date ON entrainement_incremental(date_debut);