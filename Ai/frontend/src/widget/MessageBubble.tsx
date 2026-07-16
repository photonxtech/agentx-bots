import { useState } from 'react'
import { Box, IconButton, Stack, Tooltip, Typography } from '@mui/material'
import ContentCopyIcon from '@mui/icons-material/ContentCopy'
import ThumbUpOutlinedIcon from '@mui/icons-material/ThumbUpOutlined'
import ThumbDownOutlinedIcon from '@mui/icons-material/ThumbDownOutlined'
import ReplayIcon from '@mui/icons-material/Replay'
import ReactMarkdown from 'react-markdown'
import { apiClient } from '../api/client'
import type { MessageSource } from '../api/types'
import { useThemeMode } from '../context/ThemeModeContext'
import { tokens } from '../theme/tokens'

interface Props {
  id: number | 'streaming'
  role: 'user' | 'assistant'
  content: string
  sources?: MessageSource[]
  confidence?: number
  feedback?: 'up' | 'down' | null
  onRegenerate?: () => void
}

export default function MessageBubble({ id, role, content, feedback, onRegenerate }: Props) {
  const [localFeedback, setLocalFeedback] = useState(feedback ?? null)
  const { mode } = useThemeMode()
  const t = tokens[mode]
  const isUser = role === 'user'

  const submitFeedback = async (value: 'up' | 'down') => {
    if (typeof id !== 'number') return
    setLocalFeedback((prev) => (prev === value ? null : value))
    await apiClient.post(`/messages/${id}/feedback`, { feedback: value })
  }

  if (isUser) {
    return (
      <Box sx={{ display: 'flex', justifyContent: 'flex-end', mb: 2 }}>
        <Box
          sx={{
            maxWidth: '78%', px: 2, py: 1.1, borderRadius: '18px 18px 4px 18px',
            bgcolor: t.filament, color: mode === 'dark' ? '#1A1310' : '#FFFFFF',
          }}
        >
          <Typography sx={{ fontSize: 14, lineHeight: 1.5 }}>{content}</Typography>
        </Box>
      </Box>
    )
  }

  const isTyping = content === '···'

  return (
    <Box sx={{ display: 'flex', mb: 2 }}>
      <Box
        sx={{
          maxWidth: '82%', px: 2, py: isTyping ? 1.3 : 1.1, borderRadius: '18px 18px 18px 4px',
          bgcolor: t.bubble, color: 'text.primary',
        }}
      >
        {isTyping ? (
          <TypingDots color={t.ash} />
        ) : (
          <Typography
            sx={{
              fontSize: 14, lineHeight: 1.6, textAlign: 'left',
              '& p': { m: 0, mb: 1 }, '& p:last-of-type': { mb: 0 },
              '& code': { fontFamily: '"IBM Plex Mono", monospace', fontSize: 13, bgcolor: 'action.hover', px: 0.6, py: 0.1, borderRadius: 0.5 },
              '& pre code': { display: 'block', p: 1, overflowX: 'auto' },
            }}
          >
            <ReactMarkdown>{content}</ReactMarkdown>
          </Typography>
        )}

        {!isTyping && (
          <Stack direction="row" spacing={0.25} sx={{ mt: 0.25, ml: -0.5, opacity: 0.65 }}>
            <Tooltip title="Copy">
              <IconButton size="small" onClick={() => navigator.clipboard.writeText(content)}>
                <ContentCopyIcon sx={{ fontSize: 14 }} />
              </IconButton>
            </Tooltip>
            <Tooltip title="Good response">
              <IconButton size="small" color={localFeedback === 'up' ? 'primary' : 'default'} onClick={() => submitFeedback('up')}>
                <ThumbUpOutlinedIcon sx={{ fontSize: 14 }} />
              </IconButton>
            </Tooltip>
            <Tooltip title="Bad response">
              <IconButton size="small" color={localFeedback === 'down' ? 'error' : 'default'} onClick={() => submitFeedback('down')}>
                <ThumbDownOutlinedIcon sx={{ fontSize: 14 }} />
              </IconButton>
            </Tooltip>
            {onRegenerate && (
              <Tooltip title="Regenerate">
                <IconButton size="small" onClick={onRegenerate}>
                  <ReplayIcon sx={{ fontSize: 14 }} />
                </IconButton>
              </Tooltip>
            )}
          </Stack>
        )}
      </Box>
    </Box>
  )
}

function TypingDots({ color }: { color: string }) {
  return (
    <Stack direction="row" spacing={0.5} sx={{ py: 0.3 }}>
      {[0, 1, 2].map((i) => (
        <Box
          key={i}
          sx={{
            width: 6, height: 6, borderRadius: '50%', bgcolor: color,
            animation: 'typing-dot 1.2s ease-in-out infinite',
            animationDelay: `${i * 0.15}s`,
            '@keyframes typing-dot': {
              '0%, 60%, 100%': { opacity: 0.3, transform: 'translateY(0)' },
              '30%': { opacity: 1, transform: 'translateY(-2px)' },
            },
          }}
        />
      ))}
    </Stack>
  )
}
