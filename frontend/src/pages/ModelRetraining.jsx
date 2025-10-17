import React, { useState, useEffect } from 'react';
import { FaSync, FaChartLine, FaDatabase, FaHistory, FaInfoCircle, FaExclamationTriangle } from 'react-icons/fa';
import './ModelRetraining.css';

const ModelRetraining = () => {
    const [state, setState] = useState({
        modelStatus: null,
        loading: true,
        retraining: false,
        message: '',
        error: null,
        progress: null,
        classDistribution: null,
        showAdvanced: false,
        sseConnected: false,
        trainingMetrics: null,
        minSamples: 2,
        problematicClasses: {},
        // Nouveaux états pour l'entraînement incrémental
        incrementalTrainingStatus: null,
        incrementalTrainingConditions: null,
        incrementalTrainingInProgress: false,
        incrementalTrainingHistory: [],
        showIncrementalSection: true
    });

    useEffect(() => {
        fetchModelStatus();
        checkDataAvailability();
        checkIncrementalTrainingStatus();
        fetchIncrementalTrainingHistory();
        return () => setState(prev => ({...prev, progress: null}));
    }, []);

    const fetchModelStatus = async () => {
        try {
            setState(prev => ({...prev, loading: true, error: null}));
            const response = await fetch('http://localhost:5000/admin/model-status', {
                credentials: 'include'
            });
            
            const data = await response.json();
            if (!response.ok) throw new Error(data.error || 'Failed to fetch model status');
            
            setState(prev => ({...prev, modelStatus: data}));
        } catch (err) {
            setState(prev => ({
                ...prev, 
                error: {
                    title: 'Erreur de chargement',
                    details: err.message,
                    type: 'FETCH_ERROR'
                }
            }));
        } finally {
            setState(prev => ({...prev, loading: false}));
        }
    };

    const checkDataAvailability = async () => {
        try {
            setState(prev => ({...prev, loading: true}));
            const response = await fetch(
                `http://localhost:5000/admin/check-training-data?min_samples=${state.minSamples}`, 
                {credentials: 'include'}
            );
            
            if (!response.ok) {
                const errorData = await response.json();
                throw new Error(errorData.error || 'Échec de la vérification des données');
            }
            
            const data = await response.json();
            setState(prev => ({
                ...prev,
                classDistribution: data.class_distribution,
                problematicClasses: data.problematic_classes || {}
            }));
        } catch (err) {
            setState(prev => ({
                ...prev,
                error: {
                    title: 'Erreur de données',
                    details: err.message,
                    type: 'DATA_ERROR'
                }
            }));
        } finally {
            setState(prev => ({...prev, loading: false}));
        }
    };
    const checkIncrementalTrainingStatus = async () => {
        try {
            setState(prev => ({ ...prev, loading: true }));
            const response = await fetch('http://localhost:5000/admin/incremental-training/check', {
                credentials: 'include'
            });
            
            if (!response.ok) {
                const errorData = await response.json();
                throw new Error(errorData.error || 'Échec de la vérification des conditions d\'entraînement incrémental');
            }
            
            const data = await response.json();
            
            // Extraire les classes manquantes du message (si applicable)
            let missingClasses = [];
            if (!data.ready && data.message && data.message.includes('Classes manquantes:')) {
                const missingClassesStr = data.message.split('Classes manquantes:')[1].trim();
                missingClasses = missingClassesStr.split(', ');
            }
            
            setState(prev => ({
                ...prev,
                incrementalTrainingConditions: {
                    ...data,
                    missing_classes: missingClasses
                },
                incrementalTrainingInProgress: data.training_in_progress || false
            }));
        } catch (err) {
            console.error('Erreur lors de la vérification des conditions:', err);
        } finally {
            setState(prev => ({ ...prev, loading: false }));
        }
    };

    const fetchIncrementalTrainingHistory = async () => {
        try {
            setState(prev => ({ ...prev, loading: true }));
            const response = await fetch('http://localhost:5000/admin/incremental-training/history', {
                credentials: 'include'
            });
            
            if (!response.ok) {
                const errorData = await response.json();
                throw new Error(errorData.error || 'Échec de la récupération de l\'historique');
            }
            
            const data = await response.json();
            if (data.success && data.history) {
                setState(prev => ({
                    ...prev,
                    incrementalTrainingHistory: data.history
                }));
            }
        } catch (err) {
            console.error('Erreur lors de la récupération de l\'historique:', err);
        } finally {
            setState(prev => ({ ...prev, loading: false }));
        }
    };

    const startIncrementalTraining = async () => {
        try {
            setState(prev => ({
                ...prev,
                retraining: true,
                message: '',
                error: null,
                progress: 'Lancement de l\'entraînement incrémental...'
            }));

            const response = await fetch('http://localhost:5000/admin/incremental-training/start', {
                method: 'POST',
                credentials: 'include'
            });

            const data = await response.json();
            if (!response.ok) throw new Error(data.error || 'Échec du lancement de l\'entraînement incrémental');

            setState(prev => ({
                ...prev,
                message: 'Entraînement incrémental lancé avec succès!',
                incrementalTrainingInProgress: true
            }));
            
            // Rafraîchir toutes les données
            await Promise.all([
                fetchModelStatus(),
                checkDataAvailability(),
                checkIncrementalTrainingStatus(),
                fetchIncrementalTrainingHistory()
            ]);

        } catch (err) {
            setState(prev => ({
                ...prev,
                error: {
                    title: 'Erreur d\'entraînement incrémental',
                    details: err.message,
                    type: 'INCREMENTAL_TRAINING_ERROR'
                }
            }));
        } finally {
            setState(prev => ({ ...prev, retraining: false }));
        }
    };
    const handleRetrain = async () => {
        let eventSource = null;
        
        try {
            setState(prev => ({
                ...prev,
                retraining: true,
                message: '',
                error: null,
                progress: 'Initialisation...',
                sseConnected: false,
                trainingMetrics: null
            }));

            // Vérification préalable des données
            const checkRes = await fetch(
                `http://localhost:5000/admin/check-training-data?min_samples=${state.minSamples}`,
                {credentials: 'include'}
            );
            const checkData = await checkRes.json();
            
            if (!checkRes.ok || !checkData.ready) {
                throw new Error(checkData.message || 'Données insuffisantes pour l\'entraînement');
            }

            // Configuration SSE
            eventSource = new EventSource(
                `http://localhost:5000/admin/retrain-model/progress?min_samples=${state.minSamples}`,
                {withCredentials: true}
            );

            eventSource.addEventListener('connected', (e) => {
                const data = JSON.parse(e.data);
                setState(prev => ({
                    ...prev,
                    sseConnected: true,
                    progress: data.message
                }));
            });

            eventSource.addEventListener('complete', (e) => {
                const data = JSON.parse(e.data);
                setState(prev => ({
                    ...prev,
                    progress: data.message
                }));
                if (eventSource) eventSource.close();
            });

            eventSource.onmessage = (e) => {
                try {
                    const progressData = JSON.parse(e.data);
                    setState(prev => ({
                        ...prev,
                        progress: `${progressData.message} (${progressData.progress}%)`,
                        ...(progressData.metrics && {trainingMetrics: progressData.metrics})
                    }));
                } catch (err) {
                    console.error('SSE parse error:', err);
                }
            };

            eventSource.onerror = (err) => {
                console.error('SSE error:', err);
                setState(prev => ({
                    ...prev,
                    sseConnected: false,
                    progress: 'Erreur de connexion SSE'
                }));
                if (eventSource) eventSource.close();
            };

            // Lancement du réentraînement
            const response = await fetch(
                `http://localhost:5000/admin/retrain-model?min_samples=${state.minSamples}`, 
                {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({ epochs: 5 }),
                    credentials: 'include'
                }
            );

            const data = await response.json();
            if (!response.ok) throw new Error(data.error || 'Échec du réentraînement');

            setState(prev => ({
                ...prev,
                message: `Réentraînement réussi! Précision: ${data.metrics?.accuracy?.toFixed(2) || 'N/A'}%`
            }));
            
            // Rafraîchissement des données
            await Promise.all([fetchModelStatus(), checkDataAvailability()]);

        } catch (err) {
            setState(prev => ({
                ...prev,
                error: {
                    title: 'Erreur de réentraînement',
                    details: err.message,
                    type: 'TRAINING_ERROR'
                }
            }));
            await checkDataAvailability();
        } finally {
            setState(prev => ({...prev, retraining: false, sseConnected: false}));
            if (eventSource) eventSource.close();
        }
    };

    const handleMinSamplesChange = (e) => {
        const value = Math.min(10, Math.max(1, parseInt(e.target.value) || 2));
        setState(prev => ({...prev, minSamples: value}));
    };

    const toggleAdvanced = () => {
        setState(prev => ({...prev, showAdvanced: !prev.showAdvanced}));
    };

    // Modifier la fonction canRetrain pour prendre en compte l'entraînement incrémental
    const canRetrain = () => {
        // Désactiver si un réentraînement est déjà en cours
        if (state.retraining) return false;
        
        // Désactiver si un entraînement incrémental automatique est en cours
        if (state.incrementalTrainingInProgress) return false;
        
        // Vérifier les conditions sur le nombre de corrections (minimum 200 comme dans le backend)
        if (!state.modelStatus?.available_corrections || 
            state.modelStatus.available_corrections < 200) return false;
        
        // Vérifier que toutes les classes sont présentes
        if (state.incrementalTrainingConditions?.missing_classes && 
            state.incrementalTrainingConditions.missing_classes.length > 0) return false;
            
        // Vérifier que chaque classe a suffisamment d'échantillons
        return Object.entries(state.problematicClasses || {}).every(([_, count]) => count >= state.minSamples);
    };
    const getDisabledReason = () => {
        const reasons = [];
        
        if (state.retraining) return "Un réentraînement est déjà en cours";
        
        if (state.incrementalTrainingInProgress) return "Un entraînement incrémental automatique est en cours";
        
        if (!state.modelStatus?.available_corrections) {
            return "Données non chargées";
        }
        
        if (state.modelStatus.available_corrections < 10) {
            reasons.push(`Seulement ${state.modelStatus.available_corrections} corrections (10 requises)`);
        }
        // Vérifier les classes manquantes
        if (state.incrementalTrainingConditions?.missing_classes && 
            state.incrementalTrainingConditions.missing_classes.length > 0) {
            reasons.push(`Classes manquantes: ${state.incrementalTrainingConditions.missing_classes.join(', ')}`);
        }

        Object.entries(state.problematicClasses || {}).forEach(([cls, count]) => {
            if (count < state.minSamples) {
                reasons.push(`${cls}: ${count}/${state.minSamples}`);
            }
        });
        
        return reasons.length > 0 ? reasons.join(", ") : "Prêt à lancer le réentraînement";
    };

    const renderStatValue = (value, isPercentage = false) => {
        if (state.loading) return <span className="loading-dots">Chargement</span>;
        if (value == null) return 'N/A';
        return isPercentage ? `${value.toFixed(2)}%` : value;
    };

    return (
        <div className="model-retraining-container">
            <h2><FaSync className="icon" /> Réentraînement du modèle</h2>

            {state.message && (
                <div className="alert alert-success">
                    <FaInfoCircle /> {state.message}
                </div>
            )}
            
            {state.error && (
                <div className="alert alert-danger">
                    <FaExclamationTriangle />
                    <div className="error-content">
                        <h4>{state.error.title}</h4>
                        <p>{state.error.details}</p>
                        {state.error.type === 'DATA_ERROR' && state.problematicClasses && (
                            <div className="insufficient-data-details">
                                <h5>Classes avec données insuffisantes :</h5>
                                <ul>
                                    {Object.entries(state.problematicClasses).map(([cls, count]) => (
                                        <li key={cls}>
                                            {cls}: {count} échantillon(s) (minimum {state.minSamples} requis)
                                        </li>
                                    ))}
                                </ul>
                            </div>
                        )}
                    </div>
                </div>
            )}

            {state.progress && (
                <div className="progress-message">
                    <div className="progress-bar">
                        <div 
                            className={`progress-fill ${state.sseConnected ? 'animated' : ''}`}
                            style={{ 
                                width: state.progress.includes('%') 
                                    ? state.progress.match(/(\d+)%/)[1] + '%' 
                                    : '100%'
                            }} 
                        />
                    </div>
                    <p>
                        {state.sseConnected && <span className="connection-status connected">● Connecté</span>}
                        {state.progress}
                    </p>
                    {state.trainingMetrics && (
                        <div className="training-metrics">
                            <span>Précision: {state.trainingMetrics.accuracy ? (state.trainingMetrics.accuracy * 100).toFixed(2) + '%' : 'N/A'}</span>
                            <span>Perte: {state.trainingMetrics.loss ? state.trainingMetrics.loss.toFixed(4) : 'N/A'}</span>
                        </div>
                    )}
                </div>
            )}

            <div className="stats-container">
                <div className="stat-card">
                    <FaDatabase className="stat-icon" />
                    <h3>Données disponibles</h3>
                    <p>{renderStatValue(state.modelStatus?.available_corrections)}</p>
                    <small>Minimum 200 requises</small>
                </div>

                <div className="stat-card">
                    <FaChartLine className="stat-icon" />
                    <h3>Précision actuelle</h3>
                    <p>{renderStatValue(state.modelStatus?.original_model_metrics?.accuracy, true)}</p>
                    <small>Sur jeu de validation</small>
                </div>

                <div className="stat-card">
                    <FaHistory className="stat-icon" />
                    <h3>Dernier réentraînement</h3>
                    <p>
                        {state.modelStatus?.last_retrained_metrics ? 
                         renderStatValue(state.modelStatus.last_retrained_metrics.accuracy, true) : 'Aucun'}
                    </p>
                    <small>
                        {state.modelStatus?.last_retrained_metrics?.epochs_completed && 
                         `(${state.modelStatus.last_retrained_metrics.epochs_completed} époques)`}
                    </small>
                </div>
            </div>

            <div className="action-section">
                <button 
                    onClick={handleRetrain}
                    disabled={!canRetrain()}
                    className={`retrain-btn ${state.retraining ? 'loading' : ''}`}
                    title={getDisabledReason()}
                >
                    {state.retraining ? (
                        <>
                            <span className="spinner"></span>
                            Réentraînement en cours...
                        </>
                    ) : (
                        'Lancer le réentraînement'
                    )}
                </button>
                
                <button 
                    onClick={() => Promise.all([
                        fetchModelStatus(), 
                        checkDataAvailability(),
                        checkIncrementalTrainingStatus(),
                        fetchIncrementalTrainingHistory()
                    ])}
                    disabled={state.loading || state.retraining}
                    className="refresh-btn"
                    title="Actualiser les statistiques"
                >
                    Actualiser les stats
                </button>
                <button 
                    onClick={toggleAdvanced} 
                    className="advanced-btn"
                    title="Afficher/Masquer les options avancées"
                >
                    {state.showAdvanced ? 'Masquer les options' : 'Options avancées'}
                </button>
            </div>

            {state.showAdvanced && (
                <div className="advanced-options">
                    <h4><FaInfoCircle /> Options Avancées</h4>
                    <div className="form-group">
                        <label>Seuil minimum par classe (1-10):</label>
                        <input 
                            type="number" 
                            min="1" 
                            max="10" 
                            value={state.minSamples}
                            onChange={handleMinSamplesChange}
                        />
                    </div>
                    <button 
                        onClick={checkDataAvailability} 
                        className="save-btn"
                        disabled={state.loading}
                    >
                        Appliquer les changements
                    </button>
                </div>
            )}

            <div className="info-box">
                <h4><FaInfoCircle /> Instructions :</h4>
                <ul>
                    <li>Le réentraînement nécessite au moins 200 corrections validées</li>
                    <li>Chaque type d'empreinte doit être présent dans les données</li>
                    <li>Chaque type d'empreinte doit avoir au moins {state.minSamples} échantillon(s)</li>
                    <li>Le processus peut prendre plusieurs minutes</li>
                    <li>La progression s'affiche en temps réel</li>
                    <li>Passez la souris sur le bouton désactivé pour voir la raison</li>
                </ul>
            </div>
        </div>
    );
};

export default ModelRetraining;