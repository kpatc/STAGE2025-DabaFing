import React from 'react';
import { useNavigate } from 'react-router-dom';
import './FingerprintAnalysis.css';
import logo from '../assets/images/empreinte.png';
import { FaFingerprint, FaRobot, FaChartLine, FaArrowRight } from 'react-icons/fa';
import { motion } from 'framer-motion';

const FingerprintAnalysis = () => {
  const navigate = useNavigate();

  const handleArrowClick = () => {
    navigate('/ridge-counting');
  };

  return (
    <div className="modern-analysis-container">
      <div className="modern-background-pattern"></div>
      
      <div className="modern-header-container">
        <div className="modern-logo-container">
          <motion.img 
            src={logo} 
            alt="Fingerprint Analysis Logo" 
            className="modern-logo"
            initial={{ scale: 0.9 }}
            animate={{ scale: 1 }}
            transition={{ duration: 0.5 }}
          />
          <div className="modern-logo-glow"></div>
        </div>

        <motion.h1 className="modern-analysis-header">
          <FaFingerprint className="modern-title-icon" />
          <span>Classification des Empreintes</span>
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
            <FaFingerprint className="modern-card-icon" />
            <div className="modern-icon-glow"></div>
          </div>
          <div className="modern-card-content">
            <h3>Classification Automatique</h3>
            <div className="modern-card-divider"></div>
            <p>
              Notre application permet une classification automatique des empreintes digitales avec une précision optimale, 
              grâce à des algorithmes de pointe et une analyse minutieuse des motifs.
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
            <FaRobot className="modern-card-icon" />
            <FaChartLine className="modern-card-icon" />
            <div className="modern-icon-glow"></div>
          </div>
          <div className="modern-card-content">
            <h3>Technologie Avancée</h3>
            <div className="modern-card-divider"></div>
            <p>
              La classification des empreintes identifie les motifs spécifiques (boucles, verticilles, arcs) pour faciliter 
              leur analyse. Notre solution combine intelligence artificielle et apprentissage automatique pour des résultats 
              à la fois rapides et extrêmement fiables.
            </p>
          </div>
          <div className="modern-card-accent"></div>
        </motion.div>
      </div>

      <motion.button 
        className="modern-navigation-button next"
        onClick={handleArrowClick}
        whileHover={{ scale: 1.05 }}
        whileTap={{ scale: 0.95 }}
      >
        <FaArrowRight className="modern-button-icon" />
        <span>Comptage des crêtes</span>
        <div className="modern-button-glow"></div>
      </motion.button>
    </div>
  );
};

export default FingerprintAnalysis;