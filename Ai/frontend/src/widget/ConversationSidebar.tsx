import { Box, IconButton, List, ListItemButton, ListItemText, Stack, Tooltip } from '@mui/material'
import AddIcon from '@mui/icons-material/Add'
import DeleteIcon from '@mui/icons-material/Delete'
import { useQuery, useQueryClient } from '@tanstack/react-query'
import { apiClient } from '../api/client'
import type { Conversation } from '../api/types'

export default function ConversationSidebar({
  websiteId, sessionId, activeId, onSelect, onNewChat,
}: { websiteId: number; sessionId: string; activeId: number | null; onSelect: (id: number) => void; onNewChat: () => void }) {
  const qc = useQueryClient()
  const { data } = useQuery({
    queryKey: ['conversations', websiteId, sessionId],
    queryFn: async () =>
      (await apiClient.get<Conversation[]>('/conversations', { params: { website_id: websiteId, session_id: sessionId } })).data,
  })

  const handleDelete = async (id: number) => {
    await apiClient.delete(`/conversations/${id}`)
    qc.invalidateQueries({ queryKey: ['conversations', websiteId, sessionId] })
  }

  return (
    <Box sx={{ width: 200, borderRight: 1, borderColor: 'divider', overflowY: 'auto' }}>
      <Stack direction="row" justifyContent="space-between" alignItems="center" sx={{ p: 1 }}>
        <span>Chats</span>
        <Tooltip title="New chat"><IconButton size="small" onClick={onNewChat}><AddIcon fontSize="small" /></IconButton></Tooltip>
      </Stack>
      <List dense>
        {data?.map((c) => (
          <ListItemButton key={c.id} selected={c.id === activeId} onClick={() => onSelect(c.id)}>
            <ListItemText primary={c.title} secondary={new Date(c.created_at).toLocaleDateString()} />
            <IconButton size="small" onClick={(e) => { e.stopPropagation(); handleDelete(c.id) }}><DeleteIcon fontSize="inherit" /></IconButton>
          </ListItemButton>
        ))}
      </List>
    </Box>
  )
}
