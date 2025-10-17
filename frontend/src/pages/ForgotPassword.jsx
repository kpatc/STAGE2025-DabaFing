import React, { useState } from 'react';
import './Login.css';  // Utilisation du même fichier CSS que pour la page de connexion
import { FaEnvelope } from 'react-icons/fa';
import backdiv from '../assets/images/backdiv.png';

const ForgotPassword = () => {
  const [email, setEmail] = useState('');

  const handleSubmit = (e) => {
    e.preventDefault();

    // Crée un formulaire HTML à soumettre en POST vers Flask
    const form = document.createElement('form');
    form.method = 'POST';
    form.action = 'http://127.0.0.1:5000/forgot-password';

    const emailInput = document.createElement('input');
    emailInput.type = 'hidden';
    emailInput.name = 'email';
    emailInput.value = email;
    form.appendChild(emailInput);

    document.body.appendChild(form);
    form.submit(); // ✅ Soumet vers Flask => redirection automatique
  };

  return (
    <div className="login-page">
      <div
        className="login-form-container"
        style={{
          backgroundImage: `url(${backdiv})`,
          backgroundSize: 'cover',
          backgroundPosition: 'center',
        }}
      >
        <form className="login-form" onSubmit={handleSubmit}>
          <h2 className="login-title">
            Réinitialiser votre mot de passe <FaEnvelope />
          </h2>
          <div className="input-group">
            <label className="login-label"><FaEnvelope /> EMAIL</label>
            <input
              type="email"
              placeholder="Entrez votre email"
              value={email}
              onChange={(e) => setEmail(e.target.value)}
              className="login-input"
              required
            />
          </div>
          <button type="submit" className="login-button">
            Envoyer le lien de réinitialisation
          </button>
          <p className="back-to-login">
            <a href="/login">Retour à la connexion</a>
          </p>
        </form>
      </div>
    </div>
  );
};

export default ForgotPassword;
