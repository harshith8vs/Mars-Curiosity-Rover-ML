import React from 'react';
import { BrowserRouter as Router, Routes, Route, Navigate } from 'react-router-dom';
import { Navbar } from './components/Navbar';
import { ClassificationLab } from './pages/ClassificationLab';
import { DatasetModelsPage } from './pages/DatasetModelsPage';
import { DashboardPage } from './pages/DashboardPage';

export const App: React.FC = () => {
  return (
    <Router>
      <div className="aerospace-grid-bg" style={{ minHeight: '100vh', display: 'flex', flexDirection: 'column' }}>
        <Navbar />
        <main style={{ flex: 1, overflowY: 'auto', minWidth: 0, width: '100%' }}>
          <Routes>
            <Route path="/" element={<ClassificationLab />} />
            <Route path="/classification" element={<ClassificationLab />} />
            <Route path="/dataset-models" element={<DatasetModelsPage />} />
            <Route path="/dashboard" element={<DashboardPage />} />
            <Route path="*" element={<Navigate to="/" replace />} />
          </Routes>
        </main>
      </div>
    </Router>
  );
};

export default App;
