import { useState } from 'react'
import { Box, MenuItem, Select, Typography } from '@mui/material'
import { useWebsites } from '../api/hooks'
import ChatWidget from '../widget/ChatWidget'

export default function DemoPage() {
  const { data: websites } = useWebsites()
  const [selectedId, setSelectedId] = useState<number | ''>('')

  const selected = websites?.find((w) => w.id === selectedId) ?? websites?.[0]

  return (
    <Box sx={{ p: 4 }}>
      <Typography variant="h4">Demo site</Typography>
      <Typography color="text.secondary" sx={{ mb: 2 }}>
        This page simulates a customer's website with the chat widget embedded.
      </Typography>
      {websites && websites.length > 0 && (
        <Select value={selected?.id ?? ''} onChange={(e) => setSelectedId(e.target.value as number)} size="small">
          {websites.map((w) => (
            <MenuItem key={w.id} value={w.id}>{w.name}</MenuItem>
          ))}
        </Select>
      )}
      {websites && websites.length === 0 && (
        <Typography color="text.secondary">No websites yet — add one in the admin dashboard at /admin.</Typography>
      )}
      {selected && <ChatWidget website={selected} />}
    </Box>
  )
}
