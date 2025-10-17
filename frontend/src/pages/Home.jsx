import { Link } from "react-router-dom";
import "./Home.css";
import fingerprint from "../assets/images/empreinte.png";
import { useEffect, useRef } from "react";

function Home() {
  const fingerprintRef = useRef(null);

  useEffect(() => {
    // Animation pour les lettres de BIENVENUE SUR
    const letters = document.querySelectorAll('.animated-letter');
    letters.forEach((letter, index) => {
      letter.style.animation = `letterFloat 3s ease-in-out ${index * 0.1}s infinite alternate`;
    });

    // Animation pour le sous-titre
    const subtitle = document.querySelector('.inline-subtitle');
    subtitle.style.animation = 'subtitleFadeIn 2s ease-out forwards';

    // Animation interactive pour l'empreinte digitale
    const fingerprint = fingerprintRef.current;
    if (fingerprint) {
      fingerprint.addEventListener('mousemove', handleFingerprintMove);
      fingerprint.addEventListener('mouseleave', handleFingerprintLeave);
    }

    return () => {
      if (fingerprint) {
        fingerprint.removeEventListener('mousemove', handleFingerprintMove);
        fingerprint.removeEventListener('mouseleave', handleFingerprintLeave);
      }
    };
  }, []);

  const handleFingerprintMove = (e) => {
    const { left, top, width, height } = e.target.getBoundingClientRect();
    const x = (e.clientX - left) / width;
    const y = (e.clientY - top) / height;
    
    e.target.style.transform = `perspective(1000px) rotateX(${(0.5 - y) * 15}deg) rotateY(${(x - 0.5) * 15}deg) scale(1.05)`;
    e.target.style.filter = `drop-shadow(0 0 15px rgba(110, 0, 255, 0.7)) brightness(${1 + (0.5 - y) * 0.2})`;
  };

  const handleFingerprintLeave = (e) => {
    e.target.style.transform = 'perspective(1000px) rotateX(0) rotateY(0) scale(1)';
    e.target.style.filter = 'drop-shadow(0 0 10px rgba(110, 0, 255, 0.5))';
  };

  return (
    <div className="home-container">
      <div className="top-middle-nav">
        <Link to="/about" className="nav-btn wave-effect">À PROPOS <span className="nav-icon">🌀</span></Link>
        <Link to="/contact" className="nav-btn wave-effect">CONTACT <span className="nav-icon">📞</span></Link>
      </div>

      <div className="welcome-section">
        <div className="title-image-group">
          <div className="title-and-sub">
            <div className="title-block">
              <h1 className="main-title">
                {['B', 'I', 'E', 'N', 'V', 'E', 'N', 'U', 'E'].map((letter, index) => (
                  <span key={index} className="animated-letter">{letter}</span>
                ))}
              </h1>
              <h1 className="main-title">
                {['S', 'U', 'R'].map((letter, index) => (
                  <span key={index} className="animated-letter">{letter}</span>
                ))}
              </h1>
            </div>
            <p className="inline-subtitle">
              <span className="subtitle-text">Analyse intelligente et interactive de vos empreintes digitales</span>
            </p>
          </div>
          <div className="fingerprint-container">
            <img 
              ref={fingerprintRef}
              src={fingerprint} 
              alt="Empreinte Digitale" 
              className="fingerprint-img" 
            />
            <div className="fingerprint-glow"></div>
            <div className="fingerprint-scan"></div>
          </div>
        </div>

        <div className="buttons">
          <Link to="/login" className="btn btn-primary wave-effect">SE CONNECTER <span className="btn-icon">⏎</span></Link>
          <Link to="/register" className="btn btn-secondary wave-effect">CRÉER UN COMPTE <span className="btn-icon">👤</span></Link>
        </div>
      </div>
    </div>
  );
}

export default Home;