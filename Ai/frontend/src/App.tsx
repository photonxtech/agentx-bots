import { lazy, Suspense } from 'react'
import { Navigate, Route, Routes } from 'react-router-dom'
import DemoPage from './pages/DemoPage'
import RequireAdmin from './admin/RequireAdmin'

const LoginPage = lazy(() => import('./admin/LoginPage'))
const AdminLayout = lazy(() => import('./admin/AdminLayout'))
const WebsitesPage = lazy(() => import('./admin/WebsitesPage'))
const AnalyticsPage = lazy(() => import('./admin/AnalyticsPage'))

function App() {
  return (
    <Suspense fallback={null}>
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
    </Suspense>
  )
}

export default App
