import React, { useState, useEffect } from 'react';
import axios from 'axios';
import { Link } from 'react-router-dom';
import './UserManagement.css';

const UserManagement = () => {
    const [users, setUsers] = useState([]);
    const [loading, setLoading] = useState(true);

    useEffect(() => {
        const fetchUsers = async () => {
            try {
                const response = await axios.get('http://localhost:5000/admin/users', {
                    withCredentials: true
                });
                setUsers(response.data);
                setLoading(false);
            } catch (error) {
                console.error('Error fetching users:', error);
                setLoading(false);
            }
        };

        fetchUsers();
    }, []);

    const toggleAdmin = async (userId) => {
        try {
            await axios.post(`http://localhost:5000/admin/user/${userId}/toggle-admin`, {}, {
                withCredentials: true
            });
            setUsers(users.map(user => 
                user.id_utilisateur === userId 
                    ? { ...user, role: user.role === 'admin' ? 'utilisateur' : 'admin' } 
                    : user
            ));
        } catch (error) {
            console.error('Error toggling admin status:', error);
        }
    };

    const deleteUser = async (userId) => {
        if (window.confirm('Êtes-vous sûr de vouloir supprimer cet utilisateur ?')) {
            try {
                await axios.post(`http://localhost:5000/admin/user/${userId}/delete`, {}, {
                    withCredentials: true
                });
                setUsers(users.filter(user => user.id_utilisateur !== userId));
            } catch (error) {
                console.error('Error deleting user:', error);
            }
        }
    };

    if (loading) return <div>Chargement...</div>;

    return (
        <div className="user-management">
            <h2>Gestion des utilisateurs</h2>
            <Link to="/admin" className="back-link">← Retour au tableau de bord</Link>
            
            <table className="user-table">
                <thead>
                    <tr>
                        <th>ID</th>
                        <th>Nom</th>
                        <th>Email</th>
                        <th>Rôle</th>
                        <th>Actions</th>
                    </tr>
                </thead>
                <tbody>
                    {users.map(user => (
                        <tr key={user.id_utilisateur}>
                            <td>{user.id_utilisateur}</td>
                            <td>{user.nom} {user.prenom}</td>
                            <td>{user.email}</td>
                            <td>{user.role}</td>
                            <td className="actions">
                                <button 
                                    onClick={() => toggleAdmin(user.id_utilisateur)}
                                    className={`btn ${user.role === 'admin' ? 'btn-warning' : 'btn-success'}`}
                                >
                                    {user.role === 'admin' ? 'Rétrograder' : 'Promouvoir admin'}
                                </button>
                                <button 
                                    onClick={() => deleteUser(user.id_utilisateur)}
                                    className="btn btn-danger"
                                >
                                    Supprimer
                                </button>
                            </td>
                        </tr>
                    ))}
                </tbody>
            </table>
        </div>
    );
};

export default UserManagement;