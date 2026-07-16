import { useState, type KeyboardEvent } from 'react'
import { Box, IconButton, InputBase } from '@mui/material'
import ArrowUpwardIcon from '@mui/icons-material/ArrowUpward'
import StopIcon from '@mui/icons-material/Stop'

export default function ChatInput({
  onSend, onStop, disabled,
}: { onSend: (text: string) => void; onStop: () => void; disabled: boolean }) {
  const [value, setValue] = useState('')

  const handleSend = () => {
    if (!value.trim()) return
    onSend(value.trim())
    setValue('')
  }

  const handleKeyDown = (e: KeyboardEvent<HTMLInputElement | HTMLTextAreaElement>) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      handleSend()
    }
  }

  return (
    <Box sx={{ p: 1.5 }}>
      <Box
        sx={{
          display: 'flex', alignItems: 'center', gap: 1, pl: 2, pr: 0.75, py: 0.5,
          borderRadius: 999, border: '1px solid', borderColor: 'divider', bgcolor: 'background.paper',
          transition: 'border-color 0.15s ease', '&:focus-within': { borderColor: 'primary.main' },
        }}
      >
        <InputBase
          fullWidth multiline maxRows={4} placeholder="Ask anything…" value={value}
          onChange={(e) => setValue(e.target.value)} onKeyDown={handleKeyDown} disabled={disabled}
          sx={{ fontSize: 14, '& textarea': { lineHeight: 1.5 } }}
        />
        {disabled ? (
          <IconButton
            onClick={onStop}
            size="small"
            sx={{ bgcolor: 'error.main', color: '#fff', width: 32, height: 32, '&:hover': { bgcolor: 'error.dark' } }}
          >
            <StopIcon sx={{ fontSize: 15 }} />
          </IconButton>
        ) : (
          <IconButton
            onClick={handleSend}
            disabled={!value.trim()}
            size="small"
            sx={{
              bgcolor: 'primary.main', color: 'primary.contrastText', width: 32, height: 32,
              '&:hover': { bgcolor: 'primary.main', filter: 'brightness(1.1)' },
              '&.Mui-disabled': { bgcolor: 'action.disabledBackground', color: 'action.disabled' },
            }}
          >
            <ArrowUpwardIcon sx={{ fontSize: 16 }} />
          </IconButton>
        )}
      </Box>
    </Box>
  )
}
