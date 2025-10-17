import { useState } from "react";
import { BrowserRouter as Router, Routes, Route, Navigate } from "react-router-dom";

// Pages utilisateurs
import Home from "./pages/Home";
import Login from "./pages/Login";
import Register from "./pages/Register";
import Profile from "./pages/Profile";
import ForgotPassword from "./pages/ForgotPassword";
import ConfirmationPage from "./pages/ConfirmationPage";
import FingerprintAnalysis from "./pages/FingerprintAnalysis";
import FingerprintUpload from "./pages/FingerprintUpload";
import MultiFingerprintUpload from "./pages/MultiFingerprintUpload";
import RidgeCounting from "./pages/RidgeCounting";
import AnalysisHistory from "./pages/AnalysisHistory";
import AnalysisDetail from "./pages/AnalysisDetail";
import DatasetAccess from "./pages/DatasetAccess"; // Nouvelle page pour l'accès au dataset

// Pages admin
import AdminDashboard from './pages/AdminDashboard';
import UserManagement from './pages/UserManagement';
import ModelAccuracy from './pages/ModelAccuracy';
import CorrectionManagement from './pages/CorrectionManagement';
import ModelRetraining from './pages/ModelRetraining';

function App() {
  const [user, setUser] = useState(null); // À connecter avec l'auth réelle si besoin

  return (
    <Router>
      <Routes>
        {/* Routes publiques */}
        <Route path="/" element={<Home />} />
        <Route path="/login" element={<Login setUser={setUser} />} />
        <Route path="/register" element={<Register />} />
        <Route path="/profile" element={<Profile />} />
        <Route path="/forgot-password" element={<ForgotPassword />} />
        <Route path="/confirmation" element={<ConfirmationPage />} />
        <Route path="/dataset-access" element={<DatasetAccess />} /> {/* Nouvelle route */}

        {/* Analyse empreinte digitale */}
        <Route path="/fingerprint-analysis" element={<FingerprintAnalysis />} />
        <Route path="/fingerprint-upload" element={<FingerprintUpload />} />
        <Route path="/multi-fingerprint-upload" element={<MultiFingerprintUpload />} />
        <Route path="/ridge-counting" element={<RidgeCounting />} />
        
        {/* Historique des analyses */}
        <Route path="/history" element={<AnalysisHistory />} />
        <Route path="/analysis/:id" element={<AnalysisDetail />} />

        {/* Routes admin protégées */}
        <Route 
          path="/admin" 
          element={user?.role === 'admin' ? <AdminDashboard /> : <Navigate to="/" />} 
        />
        <Route 
          path="/admin/users" 
          element={user?.role === 'admin' ? <UserManagement /> : <Navigate to="/" />} 
        />
        <Route 
          path="/admin/model-accuracy" 
          element={user?.role === 'admin' ? <ModelAccuracy /> : <Navigate to="/" />} 
        />
        <Route 
          path="/admin/corrections" 
          element={user?.role === 'admin' ? <CorrectionManagement /> : <Navigate to="/" />} 
        />
        <Route 
          path="/admin/model-retraining" 
          element={user?.role === 'admin' ? <ModelRetraining /> : <Navigate to="/" />} 
        />
      </Routes>
    </Router>
  );
}

export default App;