import { Navigate, Route, Routes } from 'react-router-dom'
import DemoPage from './pages/DemoPage'
import LoginPage from './admin/LoginPage'
import RequireAdmin from './admin/RequireAdmin'
import AdminLayout from './admin/AdminLayout'
import WebsitesPage from './admin/WebsitesPage'
import AnalyticsPage from './admin/AnalyticsPage'

function App() {
  return (
    <Routes>
      <Route path="/" element={<DemoPage />} />
      <Route path="/admin/login" element={<LoginPage />} />
      <Route
        path="/admin"
        element={
          <RequireAdmin>
            <AdminLayout />
          </RequireAdmin>
        }
      >
        <Route index element={<Navigate to="/admin/websites" replace />} />
        <Route path="websites" element={<WebsitesPage />} />
        <Route path="analytics" element={<AnalyticsPage />} />
      </Route>
    </Routes>
  )
}

export default App
