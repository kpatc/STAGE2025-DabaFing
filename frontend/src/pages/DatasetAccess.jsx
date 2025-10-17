import React from 'react';
import { 
  FaDatabase, 
  FaEnvelope, 
  FaUserTie, 
  FaUniversity, 
  FaProjectDiagram,
  FaClock,
  FaFileExport,
  FaShieldAlt
} from 'react-icons/fa';
import { FiExternalLink } from 'react-icons/fi';
import '../pages/DatasetAccess.css';

const DatasetAccess = () => {
  const email = "jean.bosco@e-polytechnique.ma";
  
  return (
    <div className="dataset-access-container">
      <div className="glass-card">
        <div className="card-header">
          <div className="header-icon-wrapper">
            <FaDatabase className="header-icon" />
          </div>
          <div className="header-content">
            <h1>Accès au Dataset Biométrique</h1>
            <p className="subtitle">Collection de données de recherche avancée</p>
          </div>
        </div>

        <div className="card-content">
          <div className="intro-section">
            <p>
              Notre base de données exclusive offre des empreintes digitales haute résolution 
              pour la recherche académique. Soumettre une demande pour accéder à l'ensemble complet.
            </p>
          </div>

          <div className="contact-section">
            <div className="contact-card">
              <div className="contact-header">
                <FaEnvelope className="icon" />
                <h3>Demande d'accès</h3>
              </div>
              <a 
                href={`mailto:${email}?subject=Demande d'accès au dataset biométrique`}
                className="email-button"
              >
                {email}
                <FiExternalLink className="external-icon" />
              </a>
              <p className="response-time">Réponse sous 48 heures</p>
            </div>
          </div>

          <div className="requirements-section">
            <h2 className="section-title">
              <span className="title-underline">Critères d'éligibilité</span>
            </h2>
            <div className="requirements-grid">
              <div className="requirement-item">
                <div className="icon-wrapper">
                  <FaUserTie className="requirement-icon" />
                </div>
                <h4>Identification professionnelle</h4>
                <p>Fournir vos informations académiques et pièce d'identité</p>
              </div>
              
              <div className="requirement-item">
                <div className="icon-wrapper">
                  <FaUniversity className="requirement-icon" />
                </div>
                <h4>Affiliation institutionnelle</h4>
                <p>Lettre de recommandation de votre établissement</p>
              </div>
              
              <div className="requirement-item">
                <div className="icon-wrapper">
                  <FaProjectDiagram className="requirement-icon" />
                </div>
                <h4>Projet de recherche</h4>
                <p>Description détaillée de votre méthodologie et objectifs</p>
              </div>
            </div>
          </div>

          <div className="specs-section">
            <h2 className="section-title">
              <span className="title-underline">Spécifications techniques</span>
            </h2>
            <div className="specs-grid">
              <div className="spec-item">
                <div className="spec-icon">
                  <FaClock />
                </div>
                <div>
                  <h4>Délai de traitement</h4>
                  <p>3-5 jours ouvrables</p>
                </div>
              </div>
              
              <div className="spec-item">
                <div className="spec-icon">
                  <FaFileExport />
                </div>
                <div>
                  <h4>Formats disponibles</h4>
                  <p>CSV, JSON, PNG (600dpi)</p>
                </div>
              </div>
              
              <div className="spec-item">
                <div className="spec-icon">
                  <FaShieldAlt />
                </div>
                <div>
                  <h4>Conditions d'utilisation</h4>
                  <p>Usage académique strictement</p>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default DatasetAccess;