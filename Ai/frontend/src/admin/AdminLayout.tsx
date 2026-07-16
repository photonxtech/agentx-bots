import { AppBar, Toolbar, Typography, Button, Box, Tabs, Tab, IconButton } from '@mui/material'
import Brightness4Icon from '@mui/icons-material/Brightness4'
import Brightness7Icon from '@mui/icons-material/Brightness7'
import { Outlet, useLocation, useNavigate } from 'react-router-dom'
import { useAuth } from '../context/AuthContext'
import { useThemeMode } from '../context/ThemeModeContext'

const TABS = [
  { label: 'Websites', path: '/admin/websites' },
  { label: 'Analytics', path: '/admin/analytics' },
]

export default function AdminLayout() {
  const { logout } = useAuth()
  const { mode, toggleMode } = useThemeMode()
  const location = useLocation()
  const navigate = useNavigate()

  const activeTab = TABS.findIndex((t) => location.pathname.startsWith(t.path))

  return (
    <Box>
      <AppBar position="static" elevation={0}>
        <Toolbar>
          <Typography variant="h6" sx={{ flexGrow: 1 }}>Chatbot Admin</Typography>
          <IconButton color="inherit" onClick={toggleMode}>
            {mode === 'dark' ? <Brightness7Icon /> : <Brightness4Icon />}
          </IconButton>
          <Button color="inherit" onClick={logout}>Logout</Button>
        </Toolbar>
        <Tabs value={activeTab === -1 ? 0 : activeTab} sx={{ px: 2 }}>
          {TABS.map((tab) => (
            <Tab key={tab.path} label={tab.label} onClick={() => navigate(tab.path)} />
          ))}
        </Tabs>
      </AppBar>
      <Box sx={{ p: 3 }}>
        <Outlet />
      </Box>
    </Box>
  )
}
