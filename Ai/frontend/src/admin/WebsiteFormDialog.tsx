import { useState, type FormEvent } from 'react'
import { Avatar, Box, Button, Dialog, DialogActions, DialogContent, DialogTitle, TextField } from '@mui/material'
import { useCreateWebsite } from '../api/hooks'

export default function WebsiteFormDialog({ open, onClose }: { open: boolean; onClose: () => void }) {
  const [name, setName] = useState('')
  const [url, setUrl] = useState('')
  const [logoUrl, setLogoUrl] = useState('')
  const createWebsite = useCreateWebsite()

  const handleSubmit = (e: FormEvent) => {
    e.preventDefault()
    createWebsite.mutate(
      { name, url, logo_url: logoUrl.trim() || undefined },
      { onSuccess: () => { setName(''); setUrl(''); setLogoUrl(''); onClose() } },
    )
  }

  return (
    <Dialog open={open} onClose={onClose} fullWidth maxWidth="sm">
      <DialogTitle>Add Website</DialogTitle>
      <DialogContent>
        <Box component="form" id="website-form" onSubmit={handleSubmit} sx={{ display: 'flex', flexDirection: 'column', gap: 2, pt: 1 }}>
          <TextField
            label="Name" placeholder="Photonx" value={name} onChange={(e) => setName(e.target.value)}
            required fullWidth autoFocus helperText="Shown as the assistant's name in the chat widget"
          />
          <TextField
            label="Website URL" placeholder="https://photonxtech.com" value={url} onChange={(e) => setUrl(e.target.value)}
            required fullWidth
          />
          <Box sx={{ display: 'flex', alignItems: 'center', gap: 1.5 }}>
            <Avatar src={logoUrl || undefined} sx={{ width: 36, height: 36 }}>
              {name ? name[0] : '?'}
            </Avatar>
            <TextField
              label="Logo URL (optional)" placeholder="https://photonxtech.com/logo.png"
              value={logoUrl} onChange={(e) => setLogoUrl(e.target.value)} fullWidth
              helperText="If set, this shows up as the assistant's avatar in the widget"
            />
          </Box>
        </Box>
      </DialogContent>
      <DialogActions>
        <Button onClick={onClose}>Cancel</Button>
        <Button type="submit" form="website-form" variant="contained" disabled={createWebsite.isPending}>
          {createWebsite.isPending ? 'Adding…' : 'Add'}
        </Button>
      </DialogActions>
    </Dialog>
  )
}
