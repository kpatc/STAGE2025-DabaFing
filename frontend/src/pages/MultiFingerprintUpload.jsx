import React, { useState, useEffect } from 'react';
import axios from 'axios';
import './MultiFingerprintUpload.css';
import { FaFingerprint, FaUpload, FaSpinner, FaInfoCircle, FaChevronRight, FaEdit, FaHistory } from 'react-icons/fa';

const MultiFingerprintUpload = () => {
    const [files, setFiles] = useState({
        rightThumb: null,
        rightIndex: null,
        rightMiddle: null,
        rightRing: null,
        rightPinky: null,
        leftThumb: null,
        leftIndex: null,
        leftMiddle: null,
        leftRing: null,
        leftPinky: null
    });
    const [isAnalyzing, setIsAnalyzing] = useState(false);
    const [error, setError] = useState('');
    const [results, setResults] = useState(null);
    const [showDetails, setShowDetails] = useState({});
    const [showCorrectionForm, setShowCorrectionForm] = useState(false);
    const [currentFinger, setCurrentFinger] = useState(null);
    const [correctionData, setCorrectionData] = useState({
        classification: '',
        nombre_cretes: '',
        commentaire: ''
    });
    const [correctionHistory, setCorrectionHistory] = useState({});
    const [showHistory, setShowHistory] = useState(null);

    const fingerLabels = {
        rightThumb: "Pouce droit",
        rightIndex: "Index droit",
        rightMiddle: "Majeur droit",
        rightRing: "Annulaire droit",
        rightPinky: "Auriculaire droit",
        leftThumb: "Pouce gauche",
        leftIndex: "Index gauche",
        leftMiddle: "Majeur gauche",
        leftRing: "Annulaire gauche",
        leftPinky: "Auriculaire gauche"
    };

    const handleFileChange = (finger, e) => {
        const selectedFile = e.target.files[0];
        
        if (selectedFile && selectedFile.size > 8 * 1024 * 1024) {
            setError(`Fichier trop volumineux (max 8MB) pour ${fingerLabels[finger]}`);
            return;
        }
        
        const allowedExtensions = ['png', 'jpg', 'jpeg', 'bmp'];
        const extension = selectedFile.name.split('.').pop().toLowerCase();
        
        if (!allowedExtensions.includes(extension)) {
            setError(`Format non supporté pour ${fingerLabels[finger]}. Utilisez PNG, JPG, JPEG ou BMP.`);
            return;
        }
        
        setError('');
        setFiles(prev => ({ ...prev, [finger]: selectedFile }));
    };

    const handleSubmit = async (e) => {
        e.preventDefault();
        
        for (const finger in files) {
            if (!files[finger]) {
                setError(`Veuillez sélectionner un fichier pour ${fingerLabels[finger]}`);
                return;
            }
        }
        
        setIsAnalyzing(true);
        setError('');
        
        try {
            const formData = new FormData();
            formData.append('files', files.rightThumb);
            formData.append('files', files.rightIndex);
            formData.append('files', files.rightMiddle);
            formData.append('files', files.rightRing);
            formData.append('files', files.rightPinky);
            formData.append('files', files.leftThumb);
            formData.append('files', files.leftIndex);
            formData.append('files', files.leftMiddle);
            formData.append('files', files.leftRing);
            formData.append('files', files.leftPinky);
            
            const response = await axios.post(
                'http://localhost:5000/fingerprint/multiple', 
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
            
            setResults(response.data.results);
            
            // Initialiser l'historique des corrections pour chaque doigt
            const history = {};
            response.data.results.forEach(result => {
                history[result.finger_name] = [];
            });
            setCorrectionHistory(history);
        } catch (err) {
            console.error("Erreur d'analyse:", err);
            setError(err.response?.data?.error || err.message || "Une erreur s'est produite");
        } finally {
            setIsAnalyzing(false);
        }
    };

    const fetchCorrectionHistory = async (fingerName) => {
        if (!results) return;
        
        const fingerResult = results.find(r => r.finger_name === fingerName);
        if (!fingerResult) return;
        
        try {
            const response = await axios.get(
                `http://localhost:5000/fingerprint/${fingerResult.id_analyse}/corrections`, 
                {
                    headers: {
                        'Authorization': `Bearer ${localStorage.getItem('token')}`
                    },
                    withCredentials: true
                }
            );
            
            if (response.data.success) {
                setCorrectionHistory(prev => ({
                    ...prev,
                    [fingerName]: response.data.corrections
                }));
            }
        } catch (err) {
            console.error('Erreur lors de la récupération des corrections:', err);
        }
    };

    const handleCorrectionClick = (fingerName) => {
        setCurrentFinger(fingerName);
        const fingerResult = results.find(r => r.finger_name === fingerName);
        setCorrectionData({
            classification: fingerResult.prediction,
            nombre_cretes: fingerResult.nombre_cretes,
            commentaire: ''
        });
        setShowCorrectionForm(true);
    };

    const handleSubmitCorrection = async (e) => {
        e.preventDefault();
        
        if (!correctionData.classification || !correctionData.nombre_cretes) {
            setError('Veuillez remplir tous les champs obligatoires');
            return;
        }

        const fingerResult = results.find(r => r.finger_name === currentFinger);
        
        try {
            const response = await axios.post(
                'http://localhost:5000/fingerprint/correction', 
                {
                    id_analyse: fingerResult.id_analyse,
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
                fetchCorrectionHistory(currentFinger);
            }
        } catch (err) {
            setError(err.response?.data?.error || 'Erreur lors de la soumission de la correction');
        }
    };

    const toggleDetails = (fingerName) => {
        setShowDetails(prev => ({
            ...prev,
            [fingerName]: !prev[fingerName]
        }));
    };

    const handleNewAnalysis = () => {
        setFiles({
            rightThumb: null,
            rightIndex: null,
            rightMiddle: null,
            rightRing: null,
            rightPinky: null,
            leftThumb: null,
            leftIndex: null,
            leftMiddle: null,
            leftRing: null,
            leftPinky: null
        });
        setResults(null);
        setError('');
        setShowDetails({});
        setShowCorrectionForm(false);
        setCorrectionHistory({});
        setShowHistory(null);
    };

    const toggleHistory = (fingerName) => {
        if (showHistory === fingerName) {
            setShowHistory(null);
        } else {
            if (!correctionHistory[fingerName] || correctionHistory[fingerName].length === 0) {
                fetchCorrectionHistory(fingerName);
            }
            setShowHistory(fingerName);
        }
    };

    return (
        <div className="multi-fingerprint-container">
            <div className="upload-header">
                <h1 className="upload-title">ANALYSE COMPLÈTE DES EMPREINTES</h1>
                <p className="upload-subtitle">Téléversez les images des 10 doigts pour une analyse complète</p>
            </div>
            
            {!results ? (
                <form onSubmit={handleSubmit} className="multi-upload-form">
                    <div className="hands-container">
                        {/* Main droite */}
                        <div className="hand-section right-hand">
                            <h3 className="hand-title">Main droite</h3>
                            {['rightThumb', 'rightIndex', 'rightMiddle', 'rightRing', 'rightPinky'].map(finger => (
                                <div key={finger} className="finger-upload">
                                    <label className="finger-label">{fingerLabels[finger]}</label>
                                    <div className="file-input-container">
                                        <input
                                            type="file"
                                            id={finger}
                                            accept=".png,.jpg,.jpeg,.bmp"
                                            onChange={(e) => handleFileChange(finger, e)}
                                            required
                                            className="file-input"
                                        />
                                        <label htmlFor={finger} className="file-label">
                                            <span className="file-name">
                                                {files[finger] ? files[finger].name : 'Sélectionner un fichier...'}
                                            </span>
                                            <span className="file-button">
                                                <FaUpload className="upload-icon" /> Parcourir
                                            </span>
                                        </label>
                                    </div>
                                </div>
                            ))}
                        </div>
                        
                        {/* Main gauche */}
                        <div className="hand-section left-hand">
                            <h3 className="hand-title">Main gauche</h3>
                            {['leftThumb', 'leftIndex', 'leftMiddle', 'leftRing', 'leftPinky'].map(finger => (
                                <div key={finger} className="finger-upload">
                                    <label className="finger-label">{fingerLabels[finger]}</label>
                                    <div className="file-input-container">
                                        <input
                                            type="file"
                                            id={finger}
                                            accept=".png,.jpg,.jpeg,.bmp"
                                            onChange={(e) => handleFileChange(finger, e)}
                                            required
                                            className="file-input"
                                        />
                                        <label htmlFor={finger} className="file-label">
                                            <span className="file-name">
                                                {files[finger] ? files[finger].name : 'Sélectionner un fichier...'}
                                            </span>
                                            <span className="file-button">
                                                <FaUpload className="upload-icon" /> Parcourir
                                            </span>
                                        </label>
                                    </div>
                                </div>
                            ))}
                        </div>
                    </div>
                    
                    <div className="file-requirements">
                        <FaInfoCircle className="info-icon" />
                        Formats acceptés: PNG, JPG, JPEG, BMP (max 8MB par fichier)
                    </div>
                    
                    {error && (
                        <div className="error-message">
                            <svg viewBox="0 0 24 24" width="20" height="20">
                                <path fill="currentColor" d="M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2zm1 15h-2v-2h2v2zm0-4h-2V7h2v6z"/>
                            </svg>
                            {error}
                        </div>
                    )}
                    
                    <button
                        type="submit"
                        disabled={isAnalyzing}
                        className={`analyze-button ${isAnalyzing ? 'analyzing' : ''}`}
                    >
                        {isAnalyzing ? (
                            <>
                                <FaSpinner className="spinner" />
                                ANALYSE EN COURS...
                            </>
                        ) : (
                            'LANCER L\'ANALYSE COMPLÈTE'
                        )}
                    </button>
                </form>
            ) : (
                <div className="multi-results-container">
                    <div className="results-header">
                        <h3>RÉSULTATS DE L'ANALYSE</h3>
                        <div className="results-actions">
                            <button onClick={handleNewAnalysis} className="new-analysis-button">
                                NOUVELLE ANALYSE
                            </button>
                        </div>
                    </div>
                    
                    <div className="results-summary">
                        <div className="hand-summary right-hand-summary">
                            <h4 className="summary-title">Main droite</h4>
                            {results.slice(0, 5).map((result, index) => (
                                <div key={index} className="finger-summary">
                                    <div className="finger-info">
                                        <div className="finger-name">{result.finger_name}</div>
                                        <div className={`finger-prediction ${result.prediction.toLowerCase()}`}>
                                            {result.prediction}
                                        </div>
                                    </div>
                                    <div className="finger-actions">
                                        <button 
                                            onClick={() => handleCorrectionClick(result.finger_name)}
                                            className="correction-button"
                                        >
                                            <FaEdit />
                                        </button>
                                        {correctionHistory[result.finger_name]?.length > 0 && (
                                            <button 
                                                onClick={() => toggleHistory(result.finger_name)}
                                                className="history-button"
                                            >
                                                <FaHistory />
                                            </button>
                                        )}
                                        <button 
                                            onClick={() => toggleDetails(result.finger_name)}
                                            className="details-button"
                                        >
                                            {showDetails[result.finger_name] ? 'CACHER' : 'DÉTAILS'} <FaChevronRight className={`chevron ${showDetails[result.finger_name] ? 'open' : ''}`} />
                                        </button>
                                    </div>
                                </div>
                            ))}
                        </div>
                        
                        <div className="hand-summary left-hand-summary">
                            <h4 className="summary-title">Main gauche</h4>
                            {results.slice(5).map((result, index) => (
                                <div key={index} className="finger-summary">
                                    <div className="finger-info">
                                        <div className="finger-name">{result.finger_name}</div>
                                        <div className={`finger-prediction ${result.prediction.toLowerCase()}`}>
                                            {result.prediction}
                                        </div>
                                    </div>
                                    <div className="finger-actions">
                                        <button 
                                            onClick={() => handleCorrectionClick(result.finger_name)}
                                            className="correction-button"
                                        >
                                            <FaEdit />
                                        </button>
                                        {correctionHistory[result.finger_name]?.length > 0 && (
                                            <button 
                                                onClick={() => toggleHistory(result.finger_name)}
                                                className="history-button"
                                            >
                                                <FaHistory />
                                            </button>
                                        )}
                                        <button 
                                            onClick={() => toggleDetails(result.finger_name)}
                                            className="details-button"
                                        >
                                            {showDetails[result.finger_name] ? 'CACHER' : 'DÉTAILS'} <FaChevronRight className={`chevron ${showDetails[result.finger_name] ? 'open' : ''}`} />
                                        </button>
                                    </div>
                                </div>
                            ))}
                        </div>
                    </div>
                    
                    {/* Détails pour chaque doigt */}
                    {results.map((result, index) => (
                        <div 
                            key={index} 
                            className={`finger-details ${showDetails[result.finger_name] ? 'visible' : ''}`}
                        >
                            <div className="detail-content">
                                <div className="fingerprint-image-container">
                                    <div className="fingerprint-glow"></div>
                                    <img 
                                        src={result.imageUrl} 
                                        alt={`Empreinte ${result.finger_name}`} 
                                        className="fingerprint-image"
                                    />
                                    <div className="fingerprint-scan"></div>
                                </div>
                                <div className="fingerprint-details">
                                    <div className="analysis-result">
                                        <div className="result-label">
                                            <svg viewBox="0 0 24 24" width="18" height="18">
                                                <path fill="currentColor" d="M12 2L4 5v6.09c0 5.05 3.41 9.76 8 10.91 4.59-1.15 8-5.86 8-10.91V5l-8-3z"/>
                                            </svg>
                                            Type de motif
                                        </div>
                                        <div className="result-value">{result.prediction}</div>
                                    </div>
                                    
                                    <div className="analysis-result">
                                        <div className="result-label">
                                            <svg viewBox="0 0 24 24" width="18" height="18">
                                                <path fill="currentColor" d="M17.66 7.93L12 2.27 6.34 7.93c-3.12 3.12-3.12 8.19 0 11.31C7.9 20.8 9.95 21.58 12 21.58c2.05 0 4.1-.78 5.66-2.34 3.12-3.12 3.12-8.19 0-11.31zM12 19.59c-1.6 0-3.11-.62-4.24-1.76C6.62 16.69 6 15.19 6 13.59s.62-3.11 1.76-4.24L12 5.1v14.49z"/>
                                            </svg>
                                            Nombre de crêtes
                                        </div>
                                        <div className="result-value">{result.nombre_cretes}</div>
                                    </div>
                                    
                                    <div className="confidence-meter-container">
                                        <div className="confidence-label">
                                            <svg viewBox="0 0 24 24" width="18" height="18">
                                                <path fill="currentColor" d="M11 15h2v2h-2zm0-8h2v6h-2zm.99-5C6.47 2 2 6.48 2 12s4.47 10 9.99 10C17.52 22 22 17.52 22 12S17.52 2 11.99 2zM12 20c-4.42 0-8-3.58-8-8s3.58-8 8-8 8 3.58 8 8-3.58 8-8 8z"/>
                                            </svg>
                                            Confiance de l'analyse
                                        </div>
                                        <div className="confidence-value">{(result.confidence * 100).toFixed(2)}%</div>
                                        <div className="confidence-meter">
                                            <div 
                                                className="confidence-fill" 
                                                style={{ width: `${result.confidence * 100}%` }}
                                            ></div>
                                        </div>
                                    </div>
                                </div>
                            </div>
                        </div>
                    ))}
                    
                    {/* Formulaire de correction */}
                    {showCorrectionForm && (
                        <div className="correction-form-overlay">
                            <div className="correction-form">
                                <h4>Correction pour {currentFinger}</h4>
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
                    
                    {/* Historique des corrections */}
                    {showHistory && (
                        <div className="history-overlay">
                            <div className="history-container">
                                <h4>Historique des corrections pour {showHistory}</h4>
                                <button 
                                    className="close-history" 
                                    onClick={() => setShowHistory(null)}
                                >
                                    ×
                                </button>
                                <div className="history-list">
                                    {correctionHistory[showHistory]?.length > 0 ? (
                                        correctionHistory[showHistory].map((correction, index) => (
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
                                        ))
                                    ) : (
                                        <div className="no-history">Aucune correction enregistrée</div>
                                    )}
                                </div>
                            </div>
                        </div>
                    )}
                </div>
            )}
            
            {/* SVG pour les gradients */}
            <svg className="defs-only">
                <defs>
                    <linearGradient id="gradient" x1="0%" y1="0%" x2="100%" y2="100%">
                        <stop offset="0%" stopColor="var(--primary-color)" />
                        <stop offset="100%" stopColor="var(--secondary-color)" />
                    </linearGradient>
                </defs>
            </svg>
        </div>
    );
};

export default MultiFingerprintUpload;