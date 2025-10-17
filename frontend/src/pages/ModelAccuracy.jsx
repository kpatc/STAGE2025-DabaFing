import React, { useState, useEffect } from 'react';
import axios from 'axios';
import { Link } from 'react-router-dom';
import './ModelAccuracy.css';

const ModelAccuracy = () => {
    const [accuracy, setAccuracy] = useState(null);
    const [loss, setLoss] = useState(null);
    const [loading, setLoading] = useState(true);

    useEffect(() => {
        const fetchModelAccuracy = async () => {
            try {
                const response = await axios.get('http://localhost:5000/admin/model-accuracy', {
                    withCredentials: true
                });
                setAccuracy(response.data.accuracy);
                setLoss(response.data.loss_value);
                setLoading(false);
            } catch (error) {
                console.error('Error fetching model accuracy:', error);
                setLoading(false);
            }
        };

        fetchModelAccuracy();
    }, []);

    if (loading) return <div>Chargement...</div>;
    if (!accuracy) return <div>Impossible de charger les données de précision</div>;

    return (
        <div className="model-accuracy">
            <h2>Précision du modèle</h2>
            <Link to="/admin" className="back-link">← Retour au tableau de bord</Link>
            
            <div className="accuracy-card">
                <h3>Performance actuelle du modèle</h3>
                <div className="progress-bar-container">
                    <div 
                        className="progress-bar" 
                        style={{ width: `${accuracy}%` }}
                    >
                        {accuracy.toFixed(2)}%
                    </div>
                </div>
                <p>Le modèle actuel a une précision de validation de <strong>{accuracy.toFixed(2)}%</strong>.</p>
                <p className="loss">Validation Loss: {loss}</p>
            </div>
        </div>
    );
};

export default ModelAccuracy;