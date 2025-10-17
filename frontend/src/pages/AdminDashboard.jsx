// src/pages/AdminDashboard.jsx
import React from 'react';
import { Link } from 'react-router-dom';
import { 
  FaUsersCog, 
  FaChartLine, 
  FaUserShield, 
  FaCheckCircle,
  FaSyncAlt 
} from 'react-icons/fa';
import '../pages/AdminDashboard.css';

const AdminDashboard = () => {
    return (
        <div className="admin-container">
            <h2>
                <FaUserShield className="admin-title-icon" />
                DabaFing administrateur
            </h2>
            <div className="admin-links">
                <Link to="/admin/users" className="admin-link">
                    <div className="glow-effect"></div>
                    <FaUsersCog className="admin-icon" />
                    Gestion des utilisateurs
                </Link>
                <Link to="/admin/model-accuracy" className="admin-link">
                    <div className="glow-effect"></div>
                    <FaChartLine className="admin-icon" />
                    Précision du modèle
                </Link>
                <Link to="/admin/corrections" className="admin-link">
                    <div className="glow-effect"></div>
                    <FaCheckCircle className="admin-icon" />
                    Gestion des corrections
                </Link>
                <Link to="/admin/model-retraining" className="admin-link">
                    <div className="glow-effect"></div>
                    <FaSyncAlt className="admin-icon" />
                    Réentraînement du modèle
                </Link>
            </div>
        </div>
    );
};

export default AdminDashboard;