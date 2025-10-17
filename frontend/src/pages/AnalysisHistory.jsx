import React, { useState, useEffect } from 'react';
import axios from 'axios';
import { Link } from 'react-router-dom';
import './AnalysisHistory.css';
import { FiArrowLeft } from 'react-icons/fi';

const AnalysisHistory = () => {
  const [analyses, setAnalyses] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');

  useEffect(() => {
    const fetchAnalyses = async () => {
      try {
        const response = await axios.get('http://localhost:5000/user/analyses', {
          headers: {
            'Authorization': `Bearer ${localStorage.getItem('token')}`
          },
          withCredentials: true
        });

        if (response.data.success) {
          setAnalyses(response.data.analyses);
        } else {
          setError('Erreur lors du chargement des analyses');
        }
      } catch (err) {
        setError(err.response?.data?.error || 'Erreur de connexion au serveur');
      } finally {
        setLoading(false);
      }
    };

    fetchAnalyses();
  }, []);

  if (loading) {
    return (
      <div className="loading-container">
        <div className="spinner"></div>
        <p>Chargement de votre historique...</p>
      </div>
    );
  }

  if (error) {
    return (
      <div className="error-container">
        <div className="error-icon">!</div>
        <p>{error}</p>
        <button onClick={() => window.location.reload()}>Réessayer</button>
      </div>
    );
  }

  return (
    <div className="history-container">
      
        
      
      
      <div className="history-header">
        <h1 className="history-title">HISTORIQUE DES ANALYSES</h1>
        <p className="history-subtitle">Consultation de vos analyses précédentes</p>
      </div>
      
      {analyses.length === 0 ? (
        <div className="empty-history">
          <p>Vous n'avez encore effectué aucune analyse.</p>
          <Link to="/upload" className="analyze-button">
            Analyser une empreinte
          </Link>
        </div>
      ) : (
        <div className="analyses-grid">
          {analyses.map((analysis) => (
            <div key={analysis.id_analyse} className="analysis-card">
              <div className="analysis-image">
                <img src={analysis.image_url} alt="Empreinte digitale" />
              </div>
              <div className="analysis-details">
                <h3>Analyse #{analysis.id_analyse}</h3>
                <p className="analysis-date">
                  {new Date(analysis.date).toLocaleDateString('fr-FR', {
                    day: 'numeric',
                    month: 'long',
                    year: 'numeric',
                    hour: '2-digit',
                    minute: '2-digit'
                  })}
                </p>
                <div className="analysis-result">
                  <span>Type: </span>
                  <strong>{analysis.classification}</strong>
                </div>
                <div className="analysis-result">
                  <span>Crêtes: </span>
                  <strong>{analysis.nombre_cretes}</strong>
                </div>
                {analysis.corrections.length > 0 && (
                  <div className="correction-badge">
                    {analysis.corrections.length} correction(s)
                  </div>
                )}
                <Link 
                  to={`/analysis/${analysis.id_analyse}`} 
                  className="view-details-button"
                >
                  Voir détails
                </Link>
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
};

export default AnalysisHistory;