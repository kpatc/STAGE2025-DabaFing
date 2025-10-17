import React from 'react';
import { useNavigate } from 'react-router-dom';
import './Profil.css';
import logo from '../assets/images/empreinte.png';

const Profile = () => {
  const navigate = useNavigate();

  return (
    <div className="profil-container">
      <div className="page-logo-container">
        <img src={logo} alt="Logo" className="page-logo" />
      </div>

      <h1 className="profil-header">Bienvenue</h1>
      <p className="profil-subtitle">
        Maintenant, vous pouvez effectuer des analyses sur des empreintes digitales
      </p>

      <div className="profil-actions">
        <button
          className="profil-button"
          onClick={() => navigate('/fingerprint-analysis')}
        >
          Suivant
        </button>
        <a className="profil-button" href="http://localhost:5173/">
          Se déconnecter
        </a>
      </div>
    </div>
  );
};

export default Profile;