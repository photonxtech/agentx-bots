import { createContext, useContext, useState, type ReactNode } from 'react'
import { setAuthToken } from '../api/client'

interface AuthContextValue {
  token: string | null
  login: (token: string) => void
  logout: () => void
}

const AuthContext = createContext<AuthContextValue | undefined>(undefined)

export function AuthProvider({ children }: { children: ReactNode }) {
  const [token, setToken] = useState<string | null>(() => {
    const stored = localStorage.getItem('admin_token')
    if (stored) setAuthToken(stored)
    return stored
  })

  const login = (newToken: string) => {
    localStorage.setItem('admin_token', newToken)
    setAuthToken(newToken)
    setToken(newToken)
  }

  const logout = () => {
    localStorage.removeItem('admin_token')
    setAuthToken(null)
    setToken(null)
  }

  return <AuthContext.Provider value={{ token, login, logout }}>{children}</AuthContext.Provider>
}

export function useAuth(): AuthContextValue {
  const ctx = useContext(AuthContext)
  if (!ctx) throw new Error('useAuth must be used within AuthProvider')
  return ctx
}
