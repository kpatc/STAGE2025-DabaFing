import React, { useState, useEffect } from 'react';
import axios from 'axios';
import { useNavigate } from 'react-router-dom';
import './FingerprintUpload.css';

const FingerprintUpload = () => {
  const [file, setFile] = useState(null);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [error, setError] = useState('');
  const [results, setResults] = useState(null);
  const [showCorrectionForm, setShowCorrectionForm] = useState(false);
  const [correctionData, setCorrectionData] = useState({
    classification: '',
    nombre_cretes: '',
    commentaire: ''
  });
  const [correctionHistory, setCorrectionHistory] = useState([]);
  const [showHistory, setShowHistory] = useState(false);
  const navigate = useNavigate();

  // Typewriter effect states
  const [titleText, setTitleText] = useState('');
  const [subtitleText, setSubtitleText] = useState('');
  const [resultsTitleText, setResultsTitleText] = useState('');
  const [patternText, setPatternText] = useState('');
  const [ridgesText, setRidgesText] = useState('');

  // Texts to animate
  const titleFullText = "ANALYSE D'EMPREINTE DIGITALE";
  const subtitleFullText = "Téléversez une image d'empreinte pour analyse";
  const resultsTitleFullText = "RÉSULTATS DE L'ANALYSE";
  const patternFullText = "MOTIF D'EMPREINTE";
  const ridgesFullText = "NOMBRE DE CRÊTES";

  // Typewriter effects
  useEffect(() => {
    let i = 0;
    const typing = setInterval(() => {
      if (i < titleFullText.length) {
        setTitleText(titleFullText.substring(0, i + 1));
        i++;
      } else {
        clearInterval(typing);
      }
    }, 100);
    return () => clearInterval(typing);
  }, []);

  useEffect(() => {
    if (results) {
      let i = 0;
      const typing = setInterval(() => {
        if (i < resultsTitleFullText.length) {
          setResultsTitleText(resultsTitleFullText.substring(0, i + 1));
          i++;
        } else {
          clearInterval(typing);
        }
      }, 100);
      return () => clearInterval(typing);
    }
  }, [results]);

  useEffect(() => {
    if (results) {
      let i = 0;
      const typing = setInterval(() => {
        if (i < patternFullText.length) {
          setPatternText(patternFullText.substring(0, i + 1));
          i++;
        } else {
          clearInterval(typing);
        }
      }, 100);
      return () => clearInterval(typing);
    }
  }, [results]);

  useEffect(() => {
    if (results) {
      let i = 0;
      const typing = setInterval(() => {
        if (i < ridgesFullText.length) {
          setRidgesText(ridgesFullText.substring(0, i + 1));
          i++;
        } else {
          clearInterval(typing);
        }
      }, 100);
      return () => clearInterval(typing);
    }
  }, [results]);

  const handleFileChange = (e) => {
    const selectedFile = e.target.files[0];
    
    if (selectedFile && selectedFile.size > 8 * 1024 * 1024) {
      setError('Fichier trop volumineux (max 8MB)');
      return;
    }
    
    const allowedExtensions = ['png', 'jpg', 'jpeg', 'bmp'];
    const extension = selectedFile.name.split('.').pop().toLowerCase();
    
    if (!allowedExtensions.includes(extension)) {
      setError('Format non supporté. Utilisez PNG, JPG, JPEG ou BMP.');
      return;
    }
    
    setError('');
    setFile(selectedFile);
  };

  const fetchCorrectionHistory = async (analyseId) => {
    if (!analyseId) return;
    
    try {
      const response = await axios.get(
        `http://localhost:5000/fingerprint/${analyseId}/corrections`, 
        {
          headers: {
            'Authorization': `Bearer ${localStorage.getItem('token')}`
          },
          withCredentials: true
        }
      );
      
      if (response.data.success) {
        setCorrectionHistory(response.data.corrections);
      }
    } catch (err) {
      console.error('Erreur lors de la récupération des corrections:', err);
    }
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    
    if (!file) {
      setError('Veuillez sélectionner un fichier');
      return;
    }
    
    setIsAnalyzing(true);
    setError('');
    
    try {
      const formData = new FormData();
      formData.append('file', file);
      
      const response = await axios.post(
        'http://localhost:5000/fingerprint', 
        formData, 
        {
          headers: {
            'Content-Type': 'multipart/form-data',
            'Authorization': `Bearer ${localStorage.getItem('token')}`
          },
          withCredentials: true
        }
      );
      
      if (response.data.error) {
        setError(response.data.error);
        return;
      }
      
      setResults({
        filename: response.data.filename,
        prediction: response.data.prediction,
        confidence: (response.data.confidence * 100).toFixed(2),
        nombre_cretes: response.data.nombre_cretes,
        imageUrl: response.data.imageUrl,
        id_analyse: response.data.id_analyse
      });
      
      fetchCorrectionHistory(response.data.id_analyse);
    } catch (err) {
      console.error("Erreur d'analyse:", err);
      setError(err.response?.data?.error || err.message || "Une erreur s'est produite");
    } finally {
      setIsAnalyzing(false);
    }
  };

  const handleSubmitCorrection = async (e) => {
    e.preventDefault();
    
    if (!correctionData.classification || !correctionData.nombre_cretes) {
      setError('Veuillez remplir tous les champs obligatoires');
      return;
    }

    try {
      const response = await axios.post(
        'http://localhost:5000/fingerprint/correction', 
        {
          id_analyse: results.id_analyse,
          classification: correctionData.classification,
          nombre_cretes: parseInt(correctionData.nombre_cretes),
          commentaire: correctionData.commentaire
        }, 
        {
          headers: {
            'Authorization': `Bearer ${localStorage.getItem('token')}`
          },
          withCredentials: true
        }
      );

      if (response.data.success) {
        setShowCorrectionForm(false);
        setCorrectionData({
          classification: '',
          nombre_cretes: '',
          commentaire: ''
        });
        fetchCorrectionHistory(results.id_analyse);
      }
    } catch (err) {
      setError(err.response?.data?.error || 'Erreur lors de la soumission de la correction');
    }
  };

  const handleNewAnalysis = () => {
    setFile(null);
    setResults(null);
    setError('');
    setShowCorrectionForm(false);
    setCorrectionHistory([]);
    setShowHistory(false);
    setResultsTitleText('');
    setPatternText('');
    setRidgesText('');
  };

  return (
    <div className="fingerprint-upload-container">
      {/* SVG Gradient Definition */}
      <svg className="defs-only">
        <defs>
          <linearGradient id="gradient" x1="0%" y1="0%" x2="100%" y2="100%">
            <stop offset="0%" stopColor="#5a00ff" />
            <stop offset="100%" stopColor="#00d1ff" />
          </linearGradient>
        </defs>
      </svg>

      <div className="upload-header">
        <h2 className="upload-title">{titleText}<span className="cursor">|</span></h2>
        <div className="upload-subtitle">{subtitleText}<span className="cursor">|</span></div>
        {!results && (
          <div className="analyze-prompt">
            ANALYSER VOS EMPREINTES POUR UNE IDENTIFICATION PRÉCISE
          </div>
        )}
      </div>
      
      {!results ? (
        <div className="upload-card">
          <div className="upload-icon">
            <svg viewBox="0 0 24 24">
              <path fill="url(#gradient)" d="M19,13H13V19H11V13H5V11H11V5H13V11H19V13Z" />
            </svg>
          </div>
          
          <form onSubmit={handleSubmit} className="upload-form">
            <div className="file-input-container">
              <input
                type="file"
                id="file"
                name="file"
                accept=".png,.jpg,.jpeg,.bmp"
                onChange={handleFileChange}
                className="file-input"
                required
              />
              <label htmlFor="file" className="file-label">
                {file ? file.name : 'CHOISIR UN FICHIER'}
                <span className="file-button">PARCOURIR</span>
              </label>
              <div className="file-requirements">Formats supportés : PNG, JPG, JPEG, BMP (max 8MB)</div>
            </div>
            
            {error && (
              <div className="error-message">
                <svg viewBox="0 0 24 24">
                  <path fill="#ff4d6d" d="M13,13H11V7H13M13,17H11V15H13M12,2A10,10 0 0,0 2,12A10,10 0 0,0 12,22A10,10 0 0,0 22,12A10,10 0 0,0 12,2Z" />
                </svg>
                {error}
              </div>
            )}
            
            <button
              type="submit"
              disabled={isAnalyzing || !file}
              className={`analyze-button ${isAnalyzing ? 'analyzing' : ''}`}
            >
              {isAnalyzing ? (
                <>
                  <span className="spinner"></span>
                  ANALYSE EN COURS...
                </>
              ) : (
                'ANALYSER EMPREINTE'
              )}
            </button>
          </form>
        </div>
      ) : (
        <div className="results-container">
          <div className="results-card">
            <div className="results-header">
              <h3>{resultsTitleText}<span className="cursor">|</span></h3>
              <div className="results-actions">
                <button 
                  onClick={() => setShowCorrectionForm(!showCorrectionForm)} 
                  className="new-analysis-button correction-button"
                >
                  <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                    <path d="M17 3a2.828 2.828 0 1 1 4 4L7.5 20.5 2 22l1.5-5.5L17 3z"></path>
                  </svg>
                  {showCorrectionForm ? 'ANNULER' : 'CORRIGER'}
                </button>
                {correctionHistory.length > 0 && (
                  <button 
                    onClick={() => setShowHistory(!showHistory)} 
                    className="history-button"
                  >
                    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                      <path d="M3 3v18h18"></path>
                      <path d="M18 17V9"></path>
                      <path d="M13 17V5"></path>
                      <path d="M8 17v-3"></path>
                    </svg>
                    {showHistory ? 'CACHER HIST.' : 'VOIR HIST.'}
                  </button>
                )}
                <button onClick={handleNewAnalysis} className="new-analysis-button">
                  <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                    <path d="M3 6h18"></path>
                    <path d="M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6m3 0V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2"></path>
                    <path d="M10 11v6"></path>
                    <path d="M14 11v6"></path>
                  </svg>
                  NOUVELLE ANALYSE
                </button>
              </div>
            </div>
            
            {showCorrectionForm && (
              <div className="correction-form-overlay">
                <div className="correction-form">
                  <h4>Proposer une correction</h4>
                  <form onSubmit={handleSubmitCorrection}>
                    <div className="form-group">
                      <label>Type de motif:</label>
                      <select
                        value={correctionData.classification}
                        onChange={(e) => setCorrectionData({
                          ...correctionData, 
                          classification: e.target.value
                        })}
                        required
                      >
                        <option value="">Sélectionnez...</option>
                        {[
                          'Accidental',
                          'Ambiguous',
                          'Simple Arch', 
                          'Tented Arch',
                          'Tented Arch with Loop',
                          'Ulnar/Radial Loop',
                          'Target Centric Whorl',
                          'Elongated Target Centric Whorl',
                          'Spiral Whorl',
                          'Elongated Spiral Whorl', 
                          'Ulnar/Radial Peacock Eye Whorl',
                          'Double Loop'
                        ].map(type => (
                          <option key={type} value={type}>{type}</option>
                        ))}
                      </select>
                    </div>
                    
                    <div className="form-group">
                      <label>Nombre de crêtes:</label>
                      <input
                        type="number"
                        value={correctionData.nombre_cretes}
                        onChange={(e) => setCorrectionData({
                          ...correctionData, 
                          nombre_cretes: e.target.value
                        })}
                        min="0"
                        required
                      />
                    </div>
                    
                    <div className="form-group">
                      <label>Commentaire (optionnel):</label>
                      <textarea
                        value={correctionData.commentaire}
                        onChange={(e) => setCorrectionData({
                          ...correctionData, 
                          commentaire: e.target.value
                        })}
                        placeholder="Ajoutez des détails sur la correction proposée..."
                      />
                    </div>
                    
                    <div className="correction-form-buttons">
                      <button 
                        type="button" 
                        onClick={() => setShowCorrectionForm(false)} 
                        className="cancel-button"
                      >
                        ANNULER
                      </button>
                      <button type="submit" className="submit-correction-button">
                        SOUMETTRE
                      </button>
                    </div>
                  </form>
                </div>
              </div>
            )}
            
            {showHistory && (
              <div className="history-overlay">
                <div className="history-container">
                  <h4>Historique des corrections</h4>
                  <button 
                    className="close-history" 
                    onClick={() => setShowHistory(false)}
                  >
                    ×
                  </button>
                  <div className="history-list">
                    {correctionHistory.map((correction, index) => (
                      <div key={index} className={`history-item ${correction.statut}`}>
                        <div className="history-header">
                          <span className="history-user">{correction.user}</span>
                          <span className="history-date">
                            {new Date(correction.date).toLocaleString()}
                          </span>
                          <span className={`history-status ${correction.statut}`}>
                            {correction.statut === 'en_attente' ? 'EN ATTENTE' : 
                             correction.statut === 'validee' ? 'VALIDÉE' : 'REJETÉE'}
                          </span>
                        </div>
                        <div className="history-details">
                          <div>
                            <span>Type proposé: </span>
                            <strong>{correction.classification}</strong>
                          </div>
                          <div>
                            <span>Crêtes proposées: </span>
                            <strong>{correction.nombre_cretes}</strong>
                          </div>
                          {correction.commentaire && (
                            <div className="history-comment">
                              <span>Commentaire: </span>
                              <em>{correction.commentaire}</em>
                            </div>
                          )}
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              </div>
            )}
            
            <div className="results-content">
              <div className="fingerprint-image-container">
                <div className="fingerprint-glow"></div>
                <img 
                  src={results.imageUrl} 
                  alt="Analyzed fingerprint" 
                  className="fingerprint-image"
                />
                <div className="fingerprint-scan"></div>
                <div className="filename">{results.filename}</div>
              </div>
              
              <div className="results-details">
                <div className="analysis-result">
                  <div className="result-label">
                    <svg viewBox="0 0 24 24" width="18" height="18">
                      <path fill="#5a00ff" d="M12,2A10,10 0 0,0 2,12A10,10 0 0,0 12,22A10,10 0 0,0 22,12A10,10 0 0,0 12,2M7.07,18.28C7.5,17.38 10.12,16.5 12,16.5C13.88,16.5 16.5,17.38 16.93,18.28C15.57,19.36 13.86,20 12,20C10.14,20 8.43,19.36 7.07,18.28M18.36,16.83C16.93,15.09 13.46,14.5 12,14.5C10.54,14.5 7.07,15.09 5.64,16.83C4.62,15.5 4,13.82 4,12C4,7.59 7.59,4 12,4C16.41,4 20,7.59 20,12C20,13.82 19.38,15.5 18.36,16.83M12,6C10.06,6 8.5,7.56 8.5,9.5C8.5,11.44 10.06,13 12,13C13.94,13 15.5,11.44 15.5,9.5C15.5,7.56 13.94,6 12,6M12,11A1.5,1.5 0 0,1 10.5,9.5A1.5,1.5 0 0,1 12,8A1.5,1.5 0 0,1 13.5,9.5A1.5,1.5 0 0,1 12,11Z" />
                    </svg>
                    {patternText}<span className="cursor">|</span>
                  </div>
                  <div className="result-value">{results.prediction}</div>
                </div>
                
                <div className="analysis-result">
                  <div className="result-label">
                    <svg viewBox="0 0 24 24" width="18" height="18">
                      <path fill="#00d1ff" d="M12,2A10,10 0 0,1 22,12A10,10 0 0,1 12,22A10,10 0 0,1 2,12A10,10 0 0,1 12,2M12,4A8,8 0 0,0 4,12A8,8 0 0,0 12,20A8,8 0 0,0 20,12A8,8 0 0,0 12,4M12,10.5A1.5,1.5 0 0,1 13.5,12A1.5,1.5 0 0,1 12,13.5A1.5,1.5 0 0,1 10.5,12A1.5,1.5 0 0,1 12,10.5M7.5,10.5A1.5,1.5 0 0,1 9,12A1.5,1.5 0 0,1 7.5,13.5A1.5,1.5 0 0,1 6,12A1.5,1.5 0 0,1 7.5,10.5M16.5,10.5A1.5,1.5 0 0,1 18,12A1.5,1.5 0 0,1 16.5,13.5A1.5,1.5 0 0,1 15,12A1.5,1.5 0 0,1 16.5,10.5Z" />
                    </svg>
                    {ridgesText}<span className="cursor">|</span>
                  </div>
                  <div className="result-value">{results.nombre_cretes}</div>
                </div>
                
                <div className="confidence-meter-container">
                  <div className="confidence-label">
                    <svg viewBox="0 0 24 24" width="18" height="18">
                      <path fill="#5a00ff" d="M12,2L4,5V11.09C4,16.14 7.41,20.85 12,22C16.59,20.85 20,16.14 20,11.09V5L12,2M12,4.15L18,6.54V11.09C18,15.09 15.45,18.79 12,19.92C8.55,18.79 6,15.1 6,11.09V6.54L12,4.15M12,7C10.34,7 9,8.34 9,10C9,11.31 9.84,12.42 11,12.83V16H13V12.83C14.16,12.42 15,11.31 15,10C15,8.34 13.66,7 12,7M12,9C12.55,9 13,9.45 13,10C13,10.55 12.55,11 12,11C11.45,11 11,10.55 11,10C11,9.45 11.45,9 12,9Z" />
                    </svg>
                    NIVEAU DE CONFIANCE: {results.confidence}%
                  </div>
                  <div className="confidence-meter">
                    <div 
                      className="confidence-fill" 
                      style={{ width: `${results.confidence}%` }}
                    ></div>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default FingerprintUpload;