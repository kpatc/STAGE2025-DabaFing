import React, { useState, useEffect } from 'react';
import './Register.css';
import { FaUser, FaEnvelope, FaKey, FaUserPlus } from 'react-icons/fa';
import backdiv from '../assets/images/backdiv.png';
import logo from '../assets/images/empreinte.png';  // Importer le logo

function Register() {
  const [formData, setFormData] = useState({
    nom: '',
    prenom: '',
    email: '',
    mot_de_passe: '',
  });

  useEffect(() => {
    document.body.classList.add('body-login');
    return () => {
      document.body.classList.remove('body-login');
    };
  }, []);

  const handleSubmit = async (e) => {
    e.preventDefault();

    const motDePasse = formData.mot_de_passe;

    // Vérification du mot de passe
    const motDePasseValide = 
      motDePasse.length >= 8 &&
      /[A-Z]/.test(motDePasse) &&     // au moins une majuscule
      /[a-z]/.test(motDePasse) &&     // au moins une minuscule
      /[!@#$%^&*(),.?":{}|<>]/.test(motDePasse); // au moins un caractère spécial

    if (!motDePasseValide) {
      alert('Le mot de passe doit contenir au moins 8 caractères, une majuscule, une minuscule et un caractère spécial.');
      return; // Arrêter l'envoi si le mot de passe n'est pas valide
    }

    try {
      const response = await fetch('http://localhost:5000/register', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(formData),
      });

      if (response.ok) {
        window.location.href = '/login';
      } else {
        const errorData = await response.json();
        alert(errorData.message || 'Une erreur est survenue');
      }
    } catch (error) {
      console.error('Erreur de soumission:', error);
      alert('Une erreur est survenue.');
    }
  };

  const handleLoginRedirect = () => {
    window.location.href = 'http://localhost:5173/login';
  };

  const animateText = (text) => {
    return text.split('').map((char, index) => (
      <span key={index} style={{ animationDelay: `${index * 0.1}s` }}>
        {char}
      </span>
    ));
  };

  return (
    <div className="login-page">
      <div className="page-logo-container">
        <img src={logo} alt="Logo" className="page-logo" />
        <span className="brand-name">Espace Inscription</span>
      </div>

      <div
        className="login-form-container"
        style={{
          backgroundImage: `url(${backdiv})`,
          backgroundSize: 'cover',
          backgroundPosition: 'center',
          position: 'relative',
          zIndex: 1,
        }}
      >
        <form className="login-form" onSubmit={handleSubmit}>
          <h2 className="login-title">
            {animateText('CREATION DE COMPTE')} <FaUserPlus />
          </h2>
          
          <div className="input-group">
            <label className="login-label"><FaUser /> NOM</label>
            <input
              type="text"
              name="nom"
              value={formData.nom}
              onChange={(e) => setFormData({ ...formData, nom: e.target.value })}
              className="login-input"
              required
            />
          </div>

          <div className="input-group">
            <label className="login-label"><FaUser /> PRENOM</label>
            <input
              type="text"
              name="prenom"
              value={formData.prenom}
              onChange={(e) => setFormData({ ...formData, prenom: e.target.value })}
              className="login-input"
              required
            />
          </div>

          <div className="input-group">
            <label className="login-label"><FaEnvelope /> EMAIL</label>
            <input
              type="email"
              name="email"
              value={formData.email}
              onChange={(e) => setFormData({ ...formData, email: e.target.value })}
              className="login-input"
              required
            />
          </div>

          <div className="input-group">
            <label className="login-label"><FaKey /> MOT DE PASSE</label>
            <input
              type="password"
              name="mot_de_passe"
              value={formData.mot_de_passe}
              onChange={(e) => setFormData({ ...formData, mot_de_passe: e.target.value })}
              className="login-input"
              required
            />
          </div>

          <button type="submit" className="login-button">
            <span className="button-text">CREER LE COMPTE</span> <FaUserPlus className="button-icon" />
          </button>

          {/* Nouveau bouton pour se connecter */}
          <div style={{ marginTop: '20px', textAlign: 'center' }}>
            <p style={{ color: 'white' }}>Si tu as déjà un compte :</p>
            <button 
              type="button" 
              onClick={handleLoginRedirect} 
              className="login-button" 
              style={{ backgroundColor: '#4CAF50', marginTop: '10px' }}
            >
              Connecte-toi
            </button>
          </div>

        </form>
      </div>
    </div>
  );
}

export default Register;
