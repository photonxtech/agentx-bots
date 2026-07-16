import type { ReactNode } from 'react'
import { Navigate } from 'react-router-dom'
import { useAuth } from '../context/AuthContext'

export default function RequireAdmin({ children }: { children: ReactNode }) {
  const { token } = useAuth()
  if (!token) return <Navigate to="/admin/login" replace />
  return <>{children}</>
}
