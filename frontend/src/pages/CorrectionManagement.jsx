import React, { useState, useEffect } from 'react';
import { Table, Button, Modal, message, Tag, Card, Divider } from 'antd';
import { 
  CheckCircleOutlined, 
  CloseCircleOutlined, 
  ArrowRightOutlined,
  FileTextOutlined,
  UserOutlined,
  ClockCircleOutlined,
  BarcodeOutlined,
  EyeOutlined
} from '@ant-design/icons';
import axios from 'axios';
import '../pages/AdminDashboard.css';

const CorrectionManagement = () => {
  const [corrections, setCorrections] = useState([]);
  const [loading, setLoading] = useState(true);
  const [selectedCorrection, setSelectedCorrection] = useState(null);
  const [isModalVisible, setIsModalVisible] = useState(false);

  useEffect(() => {
    fetchPendingCorrections();
  }, []);

  const fetchPendingCorrections = async () => {
    try {
      setLoading(true);
      const response = await axios.get('/fingerprint/corrections/pending');
      setCorrections(response.data.corrections);
      setLoading(false);
    } catch (error) {
      message.error('Échec de la récupération des corrections');
      setLoading(false);
    }
  };

  const handleReview = (correction) => {
    setSelectedCorrection(correction);
    setIsModalVisible(true);
  };

  const handleDecision = async (action) => {
    try {
      await axios.post(`/admin/correction/${selectedCorrection.id}/review`, {
        action: action
      });
      message.success(`Correction ${action === 'validate' ? 'validée' : 'rejetée'} avec succès`);
      setIsModalVisible(false);
      fetchPendingCorrections();
    } catch (error) {
      message.error(`Échec de ${action === 'validate' ? 'la validation' : 'du rejet'} de la correction`);
    }
  };

  const columns = [
    {
      title: 'ID Analyse',
      dataIndex: 'id_analyse',
      key: 'id_analyse',
      render: (text) => <span className="data-cell id-cell">#{text}</span>
    },
    {
      title: 'Utilisateur',
      dataIndex: 'user_email',
      key: 'user',
      render: (text) => (
        <div className="user-cell">
          <UserOutlined className="icon" />
          <span>{text}</span>
        </div>
      )
    },
    {
      title: 'Classification',
      dataIndex: 'classification_corrigee',
      key: 'classification',
      render: (text, record) => (
        <div className="comparison-cell">
          <Tag className="original-tag">{record.original_classification}</Tag>
          <ArrowRightOutlined className="arrow-icon" />
          <Tag className="proposed-tag" color="processing">{text}</Tag>
        </div>
      ),
    },
    {
      title: 'Crêtes',
      dataIndex: 'nombre_cretes_corrige',
      key: 'ridges',
      render: (text, record) => (
        <div className="numeric-comparison">
          <span className="original">{record.original_ridges}</span>
          <ArrowRightOutlined className="arrow-icon" />
          <span className="proposed">{text}</span>
        </div>
      ),
    },
    {
      title: 'Date',
      dataIndex: 'date_correction',
      key: 'date',
      render: (date) => (
        <div className="date-cell">
          <ClockCircleOutlined className="icon" />
          {new Date(date).toLocaleString()}
        </div>
      ),
    },
    {
      title: 'Actions',
      key: 'actions',
      render: (_, record) => (
        <Button 
          type="primary" 
          ghost
          onClick={() => handleReview(record)}
          className="review-btn"
          icon={<FileTextOutlined />}
        >
          Détails
        </Button>
      ),
    },
  ];

  return (
    <div className="admin-container ultra-modern">
      <div className="admin-header-section">
        <h1 className="admin-main-title">
          <span className="title-gradient">Gestion des Corrections</span>
        </h1>
        <p className="admin-subtitle">
          Revue et validation des modifications proposées par les experts
        </p>
      </div>
      
      <Card className="correction-card-glass" bordered={false}>
        <Table 
          columns={columns} 
          dataSource={corrections} 
          loading={loading}
          rowKey="id"
          pagination={{ 
            pageSize: 10,
            showSizeChanger: false,
            className: 'modern-pagination'
          }}
          className="ultra-modern-table"
          rowClassName="table-row"
        />
      </Card>
        
      <Modal
        title={
          <div className="modal-header">
            <h3>Validation de Correction</h3>
            <p className="modal-subtitle">ID: #{selectedCorrection?.id_analyse}</p>
          </div>
        }
        open={isModalVisible}
        onCancel={() => setIsModalVisible(false)}
        footer={null}
        width={800}
        className="correction-modal-modern"
        closeIcon={<CloseCircleOutlined style={{ color: '#fff' }} />}
      >
        {selectedCorrection && (
          <div className="correction-details-modern">
            {/* Nouvelle section pour l'image de l'empreinte */}
            <div className="fingerprint-image-section">
              <div className="detail-section-header">
                <EyeOutlined className="section-icon" />
                <h4>Empreinte digitale analysée</h4>
              </div>
              <div className="image-container">
                <img 
                  src={selectedCorrection.image_url} 
                  alt="Empreinte digitale"
                  className="fingerprint-image"
                  onError={(e) => {
                    e.target.onerror = null;
                    e.target.src = '/placeholder-fingerprint.jpg';
                  }}
                />
              </div>
            </div>

            <Divider className="modern-divider" />

            <div className="detail-section-header">
              <UserOutlined className="section-icon" />
              <h4>Informations Générales</h4>
            </div>
            <div className="detail-grid-modern">
              <div className="detail-item-modern">
                <span className="detail-label">Expert</span>
                <span className="detail-value">{selectedCorrection.user_email}</span>
              </div>
              <div className="detail-item-modern">
                <span className="detail-label">Date</span>
                <span className="detail-value">
                  {new Date(selectedCorrection.date_correction).toLocaleString()}
                </span>
              </div>
              <div className="detail-item-modern">
                <span className="detail-label">ID Analyse</span>
                <span className="detail-value id-value">#{selectedCorrection.id_analyse}</span>
              </div>
            </div>

            <Divider className="modern-divider" />

            <div className="detail-section-header">
              <BarcodeOutlined className="section-icon" />
              <h4>Modifications Proposées</h4>
            </div>
            
            <div className="modification-section">
              <div className="modification-card">
                <h5 className="modification-title">Classification</h5>
                <div className="modification-comparison">
                  <div className="modification-original">
                    <span>Original</span>
                    <Tag className="status-tag original">{selectedCorrection.original_classification}</Tag>
                  </div>
                  <ArrowRightOutlined className="comparison-arrow" />
                  <div className="modification-proposed">
                    <span>Proposé</span>
                    <Tag className="status-tag proposed" color="processing">
                      {selectedCorrection.classification_corrigee}
                    </Tag>
                  </div>
                </div>
              </div>

              <div className="modification-card">
                <h5 className="modification-title">Comptage de Crêtes</h5>
                <div className="modification-comparison">
                  <div className="modification-original">
                    <span>Original</span>
                    <div className="numeric-value">{selectedCorrection.original_ridges}</div>
                  </div>
                  <ArrowRightOutlined className="comparison-arrow" />
                  <div className="modification-proposed">
                    <span>Proposé</span>
                    <div className="numeric-value proposed">{selectedCorrection.nombre_cretes_corrige}</div>
                  </div>
                </div>
              </div>
            </div>

            {selectedCorrection.commentaire && (
              <>
                <Divider className="modern-divider" />
                <div className="comment-section-modern">
                  <h5 className="comment-title">Commentaire de l'expert</h5>
                  <div className="comment-content">
                    {selectedCorrection.commentaire}
                  </div>
                </div>
              </>
            )}

            <div className="modal-actions-modern">
              <Button 
                className="reject-btn-modern"
                icon={<CloseCircleOutlined />}
                onClick={() => handleDecision('reject')}
              >
                Rejeter
              </Button>
              <Button 
                type="primary" 
                className="validate-btn-modern"
                icon={<CheckCircleOutlined />}
                onClick={() => handleDecision('validate')}
              >
                Valider
              </Button>
            </div>
          </div>
        )}
      </Modal>
    </div>
  );
};

export default CorrectionManagement;