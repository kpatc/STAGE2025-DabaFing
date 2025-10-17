import React, { useState, useEffect } from 'react';
import './Login.css';
import { FaUser, FaEnvelope, FaKey, FaSignInAlt } from 'react-icons/fa';
import backdiv from '../assets/images/backdiv.png';
import logo from '../assets/images/empreinte.png';
import { useNavigate } from 'react-router-dom';
import axios from 'axios';

const Login = ({ setUser }) => {
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [error, setError] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const navigate = useNavigate();

  // Configuration Axios pour CORS
  axios.defaults.baseURL = 'http://localhost:5000';
  axios.defaults.withCredentials = true;
  axios.defaults.headers.post['Content-Type'] = 'application/json';

  useEffect(() => {
    document.body.classList.add('body-login');
    return () => {
      document.body.classList.remove('body-login');
    };
  }, []);

  const handleLogin = async (e) => {
    e.preventDefault();
    setIsLoading(true);
    setError('');

    try {
      const response = await axios.post('/login', {
        email: email,
        mot_de_passe: password
      });

      // Stocker l'utilisateur avec son rôle
      setUser({
        email: response.data.user.email,
        id: response.data.user.id,
        role: response.data.user.role,
        token: response.data.token // Si vous utilisez JWT
      });

      // Stocker le token dans localStorage si vous utilisez JWT
      if (response.data.token) {
        localStorage.setItem('token', response.data.token);
      }

      // Rediriger selon le rôle
      if (response.data.user.role === 'admin') {
        navigate('/admin');
      } else {
        navigate('/profile');
      }

    } catch (error) {
      console.error('Erreur de connexion:', error);
      if (error.response) {
        // Erreur retournée par le serveur
        setError(error.response.data.message || 'Email ou mot de passe incorrect');
      } else if (error.request) {
        // La requête a été faite mais aucune réponse n'a été reçue
        setError('Le serveur ne répond pas. Veuillez réessayer plus tard.');
      } else {
        // Erreur lors de la configuration de la requête
        setError('Une erreur est survenue lors de la connexion.');
      }
    } finally {
      setIsLoading(false);
    }
  };

  const animateText = (text) => {
    return text.split('').map((char, index) => (
      <span key={index} style={{ animationDelay: `${index * 0.1}s` }}>
        {char}
      </span>
    ));
  };

  const handleForgotPassword = () => {
    navigate('/forgot-password');
  };

  return (
    <>
      <div className="page-logo-container">
        <img src={logo} alt="Logo" className="page-logo" />
        <span className="brand-name">Espace Connexion</span>
      </div>

      <div className="login-page">
        <div
          className="login-form-container"
          style={{
            backgroundImage: `url(${backdiv})`,
            backgroundSize: 'cover',
            backgroundPosition: 'center',
          }}
        >
          <form className="login-form" onSubmit={handleLogin}>
            <h2 className="login-title">
              {animateText('CONNEXION UTILISATEUR')} <FaUser />
            </h2>
            
            {error && (
              <div className="error-message">
                {error}
              </div>
            )}
            
            <div className="input-group">
              <label className="login-label"><FaEnvelope /> EMAIL</label>
              <input
                type="email"
                value={email}
                onChange={(e) => setEmail(e.target.value)}
                className="login-input"
                required
              />
            </div>
            <div className="input-group">
              <label className="login-label"><FaKey /> MOT DE PASSE</label>
              <input
                type="password"
                value={password}
                onChange={(e) => setPassword(e.target.value)}
                className="login-input"
                required
              />
            </div>
            <button 
              type="submit" 
              className="gradient-btn wave-effect"
              disabled={isLoading}
            >
              {isLoading ? (
                <span className="btn-text">CONNEXION EN COURS...</span>
              ) : (
                <>
                  <span className="btn-text">SE CONNECTER</span> 
                  <FaSignInAlt className="btn-icon" />
                </>
              )}
            </button>

            <button 
              type="button" 
              className="gradient-btn-secondary wave-effect" 
              onClick={handleForgotPassword}
              disabled={isLoading}
            >
              <span className="btn-text">Mot de passe oublié </span> 
              <span className="btn-icon">?</span>
            </button>
          </form>
        </div>
      </div>
    </>
  );
};

export default Login;