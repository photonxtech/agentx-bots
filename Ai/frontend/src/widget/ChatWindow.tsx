import { useCallback, useEffect, useRef, useState } from 'react'
import { Avatar, Box, Chip, IconButton, Paper, Stack, Tooltip, Typography } from '@mui/material'
import CloseIcon from '@mui/icons-material/Close'
import MenuIcon from '@mui/icons-material/Menu'
import DeleteOutlineIcon from '@mui/icons-material/DeleteOutline'
import SmartToyRoundedIcon from '@mui/icons-material/SmartToyRounded'
import { motion } from 'framer-motion'
import { useChatSession } from './useChatSession'
import MessageBubble from './MessageBubble'
import ChatInput from './ChatInput'
import ConversationSidebar from './ConversationSidebar'
import type { Website } from '../api/types'

const SIZE_STORAGE_KEY = 'photonx-chat-widget-size'
const MIN_WIDTH = 320
const MAX_WIDTH = 720
const MIN_HEIGHT = 420
const MAX_HEIGHT = 860
const DEFAULT_SIZE = { width: 384, height: 580 }

function loadStoredSize(): { width: number; height: number } {
  try {
    const raw = localStorage.getItem(SIZE_STORAGE_KEY)
    if (!raw) return DEFAULT_SIZE
    const parsed = JSON.parse(raw)
    return {
      width: Math.min(MAX_WIDTH, Math.max(MIN_WIDTH, parsed.width ?? DEFAULT_SIZE.width)),
      height: Math.min(MAX_HEIGHT, Math.max(MIN_HEIGHT, parsed.height ?? DEFAULT_SIZE.height)),
    }
  } catch {
    return DEFAULT_SIZE
  }
}

export default function ChatWindow({ website, onClose }: { website: Website; onClose: () => void }) {
  const { sessionId, conversationId, messages, isStreaming, sendMessage, stopStreaming, clearChat, loadConversation } =
    useChatSession(website.id)
  const [sidebarOpen, setSidebarOpen] = useState(false)
  const [size, setSize] = useState(loadStoredSize)
  const bottomRef = useRef<HTMLDivElement>(null)
  const lastQuestionRef = useRef<string | null>(null)
  const resizeStartRef = useRef<{ x: number; y: number; width: number; height: number } | null>(null)

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages])

  const handleResizeMove = useCallback((e: MouseEvent) => {
    const start = resizeStartRef.current
    if (!start) return
    // Anchored to the bottom-right corner, so dragging the top-left handle left/up
    // should grow the window — width grows as the pointer moves left, height as it moves up.
    const nextWidth = Math.min(MAX_WIDTH, Math.max(MIN_WIDTH, start.width + (start.x - e.clientX)))
    const nextHeight = Math.min(MAX_HEIGHT, Math.max(MIN_HEIGHT, start.height + (start.y - e.clientY)))
    setSize({ width: nextWidth, height: nextHeight })
  }, [])

  const handleResizeEnd = useCallback(() => {
    resizeStartRef.current = null
    document.body.style.userSelect = ''
    document.body.style.cursor = ''
    window.removeEventListener('mousemove', handleResizeMove)
    window.removeEventListener('mouseup', handleResizeEnd)
    setSize((current) => {
      localStorage.setItem(SIZE_STORAGE_KEY, JSON.stringify(current))
      return current
    })
  }, [handleResizeMove])

  const handleResizeStart = (e: React.MouseEvent) => {
    e.preventDefault()
    resizeStartRef.current = { x: e.clientX, y: e.clientY, width: size.width, height: size.height }
    document.body.style.userSelect = 'none'
    document.body.style.cursor = 'nwse-resize'
    window.addEventListener('mousemove', handleResizeMove)
    window.addEventListener('mouseup', handleResizeEnd)
  }

  useEffect(() => () => {
    window.removeEventListener('mousemove', handleResizeMove)
    window.removeEventListener('mouseup', handleResizeEnd)
  }, [handleResizeMove, handleResizeEnd])

  const handleSend = (text: string) => {
    lastQuestionRef.current = text
    sendMessage(text)
  }

  return (
    <motion.div
      className="chat-window-container"
      initial={{ opacity: 0, y: 32, scale: 0.97 }}
      animate={{ opacity: 1, y: 0, scale: 1 }}
      exit={{ opacity: 0, y: 32, scale: 0.97 }}
      transition={{ duration: 0.22, ease: 'easeOut' }}
      style={{ position: 'fixed', bottom: 96, right: 24, zIndex: 1300, width: size.width, height: size.height }}
    >
      <Paper sx={{
        position: 'relative', display: 'flex', height: '100%', borderRadius: '18px', overflow: 'hidden',
        boxShadow: '0 24px 64px rgba(0,0,0,0.3)',
      }}>
        <Box
          onMouseDown={handleResizeStart}
          sx={{
            position: 'absolute', top: 0, left: 0, width: 18, height: 18, zIndex: 1,
            cursor: 'nwse-resize',
            '&:hover': { '& > div': { borderColor: 'primary.main' } },
          }}
        >
          <Box sx={{
            position: 'absolute', top: 6, left: 6, width: 8, height: 8,
            borderTop: '2px solid', borderLeft: '2px solid', borderColor: 'divider',
            borderTopLeftRadius: 3, transition: 'border-color 0.15s ease',
          }} />
        </Box>
        {sidebarOpen && (
          <ConversationSidebar
            websiteId={website.id} sessionId={sessionId} activeId={conversationId}
            onSelect={loadConversation} onNewChat={clearChat}
          />
        )}
        <Box sx={{ display: 'flex', flexDirection: 'column', flex: 1, minWidth: 0 }}>
          <Stack direction="row" alignItems="center" spacing={1.25} sx={{ px: 2, py: 1.75, borderBottom: 1, borderColor: 'divider' }}>
            <IconButton size="small" onClick={() => setSidebarOpen((v) => !v)}>
              <MenuIcon sx={{ fontSize: 18 }} />
            </IconButton>
            <Avatar
              src={website.logo_url ?? undefined}
              sx={{ width: 32, height: 32, bgcolor: 'action.selected', color: 'primary.main' }}
            >
              <SmartToyRoundedIcon sx={{ fontSize: 18 }} />
            </Avatar>
            <Typography variant="subtitle2" sx={{ flex: 1, fontSize: 15 }} noWrap>
              {website.name} <Typography component="span" sx={{ color: 'text.secondary', fontSize: 13, fontFamily: 'inherit' }}>· AI Assistant</Typography>
            </Typography>
            <Tooltip title="Clear conversation">
              <IconButton size="small" onClick={clearChat}>
                <DeleteOutlineIcon sx={{ fontSize: 19 }} />
              </IconButton>
            </Tooltip>
            <IconButton size="small" onClick={onClose}>
              <CloseIcon sx={{ fontSize: 19 }} />
            </IconButton>
          </Stack>

          <Box sx={{ flex: 1, overflowY: 'auto', px: 2, py: 2 }}>
            {messages.length === 0 ? (
              <Box sx={{ display: 'flex', flexDirection: 'column', alignItems: 'center', textAlign: 'center', mt: 6 }}>
                <Typography sx={{ color: 'text.secondary', fontSize: 15 }}>Hey there</Typography>
                <Typography variant="h6" sx={{ fontSize: 22, mb: 2 }}>How can I help?</Typography>
                <Chip
                  clickable
                  onClick={() => handleSend(`What can ${website.name} help me with?`)}
                  label={`What can ${website.name} help me with?`}
                  icon={<Box sx={{ width: 8, height: 8, borderRadius: '50%', bgcolor: 'primary.main', ml: 1 }} />}
                  sx={{ bgcolor: 'action.hover', fontSize: 13, py: 2.25 }}
                />
              </Box>
            ) : (
              messages.map((m) => (
                <MessageBubble
                  key={m.id}
                  id={m.id}
                  role={m.role}
                  content={m.content || (isStreaming && m.id === 'streaming' ? '···' : '')}
                  sources={m.sources}
                  confidence={m.confidence}
                  feedback={m.feedback}
                  onRegenerate={
                    m.role === 'assistant' && lastQuestionRef.current
                      ? () => sendMessage(lastQuestionRef.current as string, { regenerate: true })
                      : undefined
                  }
                />
              ))
            )}
            <div ref={bottomRef} />
          </Box>

          <ChatInput onSend={handleSend} onStop={stopStreaming} disabled={isStreaming} />
        </Box>
      </Paper>
    </motion.div>
  )
}
