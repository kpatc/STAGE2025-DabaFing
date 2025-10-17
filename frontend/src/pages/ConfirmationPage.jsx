import React from 'react';
 // Vous pouvez personnaliser le style ici

const ConfirmationPage = () => {
  return (
    <div className="confirmation-page">
      <h2>Vérifiez votre boîte mail</h2>
      <p>
        Un lien de réinitialisation de mot de passe a été envoyé à votre adresse email.
        Veuillez cliquer sur le lien dans l’email pour réinitialiser votre mot de passe.
      </p>
      <p>
        <a href="/login">Retour à la connexion</a>
      </p>
    </div>
  );
};

export default ConfirmationPage;
