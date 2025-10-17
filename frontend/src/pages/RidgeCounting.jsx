import React from 'react';
import { useNavigate } from 'react-router-dom';
import './FingerprintAnalysis.css'; // Nous utilisons le même CSS
import logo from '../assets/images/empreinte.png';
import { 
  FaFingerprint, 
  FaCalculator, 
  FaChartBar, 
  FaArrowLeft, 
  FaHistory, 
  FaHands, 
  FaDatabase,
  FaArrowRight
} from 'react-icons/fa';
import { motion } from 'framer-motion';

const RidgeCounting = () => {
  const navigate = useNavigate();

  const handleBackClick = () => {
    navigate('/fingerprint-analysis');
  };

  const handleAnalyzeClick = () => {
    navigate('/fingerprint-upload');
  };

  const handleHistoryClick = () => {
    navigate('/history');
  };

  const handleCompleteAnalysisClick = () => {
    navigate('/multi-fingerprint-upload');
  };

  const handleResourcesClick = () => {
    navigate('/dataset-access');
  };

  return (
    <div className="modern-analysis-container">
      <div className="modern-background-pattern"></div>
      
      <div className="modern-header-container">
        <div className="modern-logo-container">
          <motion.img 
            src={logo} 
            alt="Ridge Counting Logo" 
            className="modern-logo"
            initial={{ scale: 0.9 }}
            animate={{ scale: 1 }}
            transition={{ duration: 0.5 }}
          />
          <div className="modern-logo-glow"></div>
        </div>

        <motion.h1 className="modern-analysis-header">
          <FaFingerprint className="modern-title-icon" />
          <span>Comptage des Crêtes Digitales</span>
          <div className="modern-header-underline"></div>
        </motion.h1>
      </div>
      
      <div className="modern-content-grid">
        <motion.div 
          className="modern-info-card"
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5, delay: 0.2 }}
        >
          <div className="modern-card-icon-container">
            <FaCalculator className="modern-card-icon" />
            <div className="modern-icon-glow"></div>
          </div>
          <div className="modern-card-content">
            <h3>Précision Algorithmique</h3>
            <div className="modern-card-divider"></div>
            <p>
                Notre technologie de pointe permet un comptage précis des crêtes digitales avec une exactitude supérieure à 93%,
              en utilisant des algorithmes avancés de vision par ordinateur et d'apprentissage automatique.
            </p>
          </div>
          <div className="modern-card-accent"></div>
        </motion.div>
        
        <motion.div 
          className="modern-info-card"
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5, delay: 0.4 }}
        >
          <div className="modern-card-icon-container dual-icons">
            <FaChartBar className="modern-card-icon" />
            <FaFingerprint className="modern-card-icon" />
            <div className="modern-icon-glow"></div>
          </div>
          <div className="modern-card-content">
            <h3>Technologie Biométrique</h3>
            <div className="modern-card-divider"></div>
            <p>
               Le comptage des crêtes est essentiel pour l'identification biométrique. Notre solution combine l'apprentissage par renforcement interactif (IRL)
              avec des retours humains (LHF) pour améliorer continuellement la précision.
            </p>
          </div>
          <div className="modern-card-accent"></div>
        </motion.div>
      </div>

      <motion.div 
        className="modern-action-grid"
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ duration: 0.5, delay: 0.6 }}
      >
        <motion.button 
          className="modern-action-button"
          onClick={handleAnalyzeClick}
          whileHover={{ scale: 1.05 }}
          whileTap={{ scale: 0.95 }}
        >
          <FaFingerprint className="modern-action-icon" />
          <span>Analyse simple</span>
          <div className="modern-button-glow"></div>
        </motion.button>
        
        <motion.button 
          className="modern-action-button"
          onClick={handleCompleteAnalysisClick}
          whileHover={{ scale: 1.05 }}
          whileTap={{ scale: 0.95 }}
        >
          <FaHands className="modern-action-icon" />
          <span>Analyse complète</span>
          <div className="modern-button-glow"></div>
        </motion.button>
        
        <motion.button 
          className="modern-action-button"
          onClick={handleHistoryClick}
          whileHover={{ scale: 1.05 }}
          whileTap={{ scale: 0.95 }}
        >
          <FaHistory className="modern-action-icon" />
          <span>Historique</span>
          <div className="modern-button-glow"></div>
        </motion.button>
        
        <motion.button 
          className="modern-action-button"
          onClick={handleResourcesClick}
          whileHover={{ scale: 1.05 }}
          whileTap={{ scale: 0.95 }}
        >
          <FaDatabase className="modern-action-icon" />
          <span>Ressources</span>
          <div className="modern-button-glow"></div>
        </motion.button>
      </motion.div>

      <motion.button 
        className="modern-navigation-button back"
        onClick={handleBackClick}
        whileHover={{ scale: 1.05 }}
        whileTap={{ scale: 0.95 }}
      >
        <FaArrowLeft className="modern-button-icon" />
        <span>Retour</span>
        <div className="modern-button-glow"></div>
      </motion.button>
    </div>
  );
};

export default RidgeCounting;