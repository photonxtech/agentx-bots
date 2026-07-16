import { Box, IconButton } from '@mui/material'
import ChatIcon from '@mui/icons-material/Chat'
import CloseIcon from '@mui/icons-material/Close'
import { motion } from 'framer-motion'

export default function FloatingButton({ open, onClick }: { open: boolean; onClick: () => void }) {
  return (
    <motion.div
      style={{ position: 'fixed', bottom: 24, right: 24, zIndex: 1300 }}
      whileHover={{ scale: 1.06 }}
      whileTap={{ scale: 0.94 }}
    >
      <Box sx={{ position: 'relative', width: 56, height: 56 }}>
        {!open && (
          <Box
            sx={{
              position: 'absolute', inset: 0, borderRadius: '50%', bgcolor: 'primary.main', opacity: 0.5,
              animation: 'beacon 2.4s ease-out infinite',
              '@media (prefers-reduced-motion: reduce)': { animation: 'none' },
              '@keyframes beacon': {
                '0%': { transform: 'scale(1)', opacity: 0.45 },
                '100%': { transform: 'scale(1.6)', opacity: 0 },
              },
            }}
          />
        )}
        <IconButton
          onClick={onClick}
          sx={{
            position: 'relative', width: 56, height: 56, bgcolor: 'primary.main', color: 'primary.contrastText',
            boxShadow: '0 10px 28px rgba(0,0,0,0.35)',
            '&:hover': { bgcolor: 'primary.main', filter: 'brightness(1.08)' },
          }}
        >
          {open ? <CloseIcon /> : <ChatIcon sx={{ fontSize: 22 }} />}
        </IconButton>
      </Box>
    </motion.div>
  )
}
