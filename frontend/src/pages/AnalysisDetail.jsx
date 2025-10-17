import React, { useState, useEffect } from 'react';
import axios from 'axios';
import { useParams, Link, useNavigate } from 'react-router-dom';
import { FiArrowLeft } from 'react-icons/fi';
import './AnalysisDetail.css';

const AnalysisDetail = () => {
  const { id } = useParams();
  const navigate = useNavigate();
  const [analysis, setAnalysis] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');
  const [correctionForm, setCorrectionForm] = useState({
    classification: '',
    nombre_cretes: '',
    commentaire: ''
  });

  useEffect(() => {
    const fetchAnalysis = async () => {
      try {
        const response = await axios.get(`http://localhost:5000/user/analyses/${id}`, {
          headers: {
            'Authorization': `Bearer ${localStorage.getItem('token')}`
          },
          withCredentials: true
        });

        if (response.data.success) {
          setAnalysis(response.data.analysis);
        } else {
          setError('Analyse non trouvée');
          navigate('/history', { replace: true });
        }
      } catch (err) {
        setError(err.response?.data?.error || 'Erreur de connexion au serveur');
        navigate('/history', { replace: true });
      } finally {
        setLoading(false);
      }
    };

    fetchAnalysis();
  }, [id, navigate]);

  const handleSubmitCorrection = async (e) => {
    e.preventDefault();
    
    if (!correctionForm.classification || !correctionForm.nombre_cretes) {
      setError('Veuillez remplir tous les champs obligatoires');
      return;
    }

    try {
      const response = await axios.post('http://localhost:5000/fingerprint/correction', {
        id_analyse: id,
        classification: correctionForm.classification,
        nombre_cretes: parseInt(correctionForm.nombre_cretes),
        commentaire: correctionForm.commentaire
      }, {
        headers: {
          'Authorization': `Bearer ${localStorage.getItem('token')}`
        },
        withCredentials: true
      });

      if (response.data.success) {
        const updatedResponse = await axios.get(`http://localhost:5000/user/analyses/${id}`, {
          headers: {
            'Authorization': `Bearer ${localStorage.getItem('token')}`
          },
          withCredentials: true
        });
        
        setAnalysis(updatedResponse.data.analysis);
        setCorrectionForm({
          classification: '',
          nombre_cretes: '',
          commentaire: ''
        });
        setError('');
      }
    } catch (err) {
      setError(err.response?.data?.error || 'Erreur lors de la soumission de la correction');
    }
  };

  if (loading) {
    return (
      <div className="loading-container">
        <div className="spinner"></div>
        <p>Chargement de l'analyse...</p>
      </div>
    );
  }

  if (error) {
    return (
      <div className="error-container">
        <div className="error-icon">!</div>
        <p>{error}</p>
        <Link to="/history" className="back-button">
          Retour à l'historique
        </Link>
      </div>
    );
  }

  return (
    <div className="analysis-detail-container">
      <Link to="/history" className="back-link">
        <FiArrowLeft /> Retour à l'historique
      </Link>
      
      <h1 className="analysis-title">DÉTAILS DE L'ANALYSE #{analysis.id_analyse}</h1>
      
      <div className="analysis-content">
        <div className="fingerprint-image-container">
          <img src={analysis.image_url} alt="Empreinte digitale" />
          <div className="fingerprint-glow"></div>
        </div>
        
        <div className="analysis-results">
          <div className="result-section">
            <h3>RÉSULTATS INITIAUX</h3>
            <div className="result-item">
              <span>Date: </span>
              <strong>
                {new Date(analysis.date).toLocaleDateString('fr-FR', {
                  day: 'numeric',
                  month: 'long',
                  year: 'numeric',
                  hour: '2-digit',
                  minute: '2-digit'
                })}
              </strong>
            </div>
            <div className="result-item">
              <span>Type d'empreinte: </span>
              <strong>{analysis.classification}</strong>
            </div>
            <div className="result-item">
              <span>Nombre de crêtes: </span>
              <strong>{analysis.nombre_cretes}</strong>
            </div>
          </div>
          
          <div className="correction-section">
            <h3>PROPOSER UNE CORRECTION</h3>
            <form onSubmit={handleSubmitCorrection}>
              <div className="form-group">
                <label>Type de motif:</label>
                <select
                  value={correctionForm.classification}
                  onChange={(e) => setCorrectionForm({
                    ...correctionForm,
                    classification: e.target.value
                  })}
                  required
                >
                  <option value="">Sélectionnez...</option>
                  <option value="Accidental">Accidental</option>
                  <option value="Ambiguous">Ambiguous</option>
                  <option value="Tented Arch">Tented Arch</option>
                  <option value="Simple Arch">Simple Arch</option>
                  <option value="Tented Arch with Loop">Tented Arch with Loop</option>
                  <option value="Ulnar/Radial Loop">Ulnar/Radial Loop</option>
                  <option value="Target Centric Whorl">Target Centric Whorl</option>
                  <option value="Ulnar/Radial Peacock Eye Whorl">Ulnar/Radial Peacock Eye Whorl</option>
                  <option value="Double Loop">Double Loop</option>
                  <option value="Elongated Spiral Whorl">Elongated Spiral Whorl</option>
                  <option value="Elongated Target Centric Whorl">Elongated Target Centric Whorl</option>
                  
            
                </select>
              </div>
              
              <div className="form-group">
                <label>Nombre de crêtes:</label>
                <input
                  type="number"
                  value={correctionForm.nombre_cretes}
                  onChange={(e) => setCorrectionForm({
                    ...correctionForm,
                    nombre_cretes: e.target.value
                  })}
                  min="0"
                  required
                />
              </div>
              
              <div className="form-group">
                <label>Commentaire (optionnel):</label>
                <textarea
                  value={correctionForm.commentaire}
                  onChange={(e) => setCorrectionForm({
                    ...correctionForm,
                    commentaire: e.target.value
                  })}
                />
              </div>
              
              {error && <div className="form-error">{error}</div>}
              
              <button type="submit" className="submit-button">
                Soumettre la correction
              </button>
            </form>
          </div>
        </div>
      </div>
      
      {analysis.corrections.length > 0 && (
        <div className="corrections-history">
          <h3>HISTORIQUE DES CORRECTIONS</h3>
          
          <div className="corrections-list">
            {analysis.corrections.map((correction, index) => (
              <div key={index} className={`correction-item ${correction.statut}`}>
                <div className="correction-header">
                  <span className="correction-date">
                    {new Date(correction.date).toLocaleDateString('fr-FR', {
                      day: 'numeric',
                      month: 'long',
                      year: 'numeric',
                      hour: '2-digit',
                      minute: '2-digit'
                    })}
                  </span>
                  <span className={`correction-status ${correction.statut}`}>
                    {correction.statut === 'en_attente' ? 'EN ATTENTE' : 
                     correction.statut === 'validee' ? 'VALIDÉE' : 'REJETÉE'}
                  </span>
                </div>
                
                <div className="correction-details">
                  <div>
                    <span>Type proposé: </span>
                    <strong>{correction.classification}</strong>
                  </div>
                  <div>
                    <span>Crêtes proposées: </span>
                    <strong>{correction.nombre_cretes}</strong>
                  </div>
                  {correction.commentaire && (
                    <div className="correction-comment">
                      <span>Commentaire: </span>
                      <p>{correction.commentaire}</p>
                    </div>
                  )}
                </div>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
};

export default AnalysisDetail;