import { useState } from 'react'
import {
  Alert, Box, Button, Chip, Dialog, DialogActions, DialogContent, DialogContentText, DialogTitle,
  IconButton, Paper, Snackbar, Stack, Table, TableBody, TableCell, TableHead, TableRow, Tooltip,
} from '@mui/material'
import AddIcon from '@mui/icons-material/Add'
import RefreshIcon from '@mui/icons-material/Refresh'
import SyncIcon from '@mui/icons-material/Sync'
import DeleteIcon from '@mui/icons-material/Delete'
import InfoIcon from '@mui/icons-material/Info'
import { useCrawlWebsite, useDeleteWebsite, useReindexWebsite, useSyncWebsite, useWebsites } from '../api/hooks'
import WebsiteFormDialog from './WebsiteFormDialog'
import WebsiteDetailDrawer from './WebsiteDetailDrawer'
import type { Website } from '../api/types'

const STATUS_COLOR: Record<Website['status'], 'success' | 'warning' | 'error'> = {
  active: 'success',
  crawling: 'warning',
  error: 'error',
}

function extractErrorMessage(error: unknown, fallback: string): string {
  if (error && typeof error === 'object' && 'response' in error) {
    const detail = (error as { response?: { data?: { detail?: string } } }).response?.data?.detail
    if (detail) return detail
  }
  return fallback
}

export default function WebsitesPage() {
  const { data: websites, isLoading } = useWebsites()
  const [formOpen, setFormOpen] = useState(false)
  const [selected, setSelected] = useState<Website | null>(null)
  const [actionError, setActionError] = useState<string | null>(null)
  const [pendingDelete, setPendingDelete] = useState<Website | null>(null)
  const crawl = useCrawlWebsite()
  const sync = useSyncWebsite()
  const reindex = useReindexWebsite()
  const remove = useDeleteWebsite()

  const confirmDelete = () => {
    if (!pendingDelete) return
    const site = pendingDelete
    setPendingDelete(null)
    remove.mutate(site.id, {
      onError: (error) => setActionError(extractErrorMessage(error, `Failed to delete "${site.name}".`)),
    })
  }

  const handleCrawl = (site: Website) =>
    crawl.mutate(site.id, { onError: (error) => setActionError(extractErrorMessage(error, `Failed to start crawl for "${site.name}".`)) })
  const handleSync = (site: Website) =>
    sync.mutate(site.id, { onError: (error) => setActionError(extractErrorMessage(error, `Failed to start sync for "${site.name}".`)) })
  const handleReindex = (site: Website) =>
    reindex.mutate(site.id, { onError: (error) => setActionError(extractErrorMessage(error, `Failed to start reindex for "${site.name}".`)) })

  return (
    <Box>
      <Stack direction="row" justifyContent="space-between" sx={{ mb: 2 }}>
        <h2>Websites</h2>
        <Button startIcon={<AddIcon />} variant="contained" onClick={() => setFormOpen(true)}>
          Add Website
        </Button>
      </Stack>
      <Paper sx={{ borderRadius: 3 }}>
        <Table>
          <TableHead>
            <TableRow>
              <TableCell>Name</TableCell>
              <TableCell>URL</TableCell>
              <TableCell>Status</TableCell>
              <TableCell>Last Synced</TableCell>
              <TableCell align="right">Actions</TableCell>
            </TableRow>
          </TableHead>
          <TableBody>
            {isLoading && (
              <TableRow><TableCell colSpan={5}>Loading…</TableCell></TableRow>
            )}
            {websites?.map((site) => (
              <TableRow key={site.id}>
                <TableCell>{site.name}</TableCell>
                <TableCell>{site.url}</TableCell>
                <TableCell><Chip size="small" label={site.status} color={STATUS_COLOR[site.status]} /></TableCell>
                <TableCell>{site.last_synced_at ? new Date(site.last_synced_at).toLocaleString() : 'Never'}</TableCell>
                <TableCell align="right">
                  <Tooltip title="Crawl"><IconButton onClick={() => handleCrawl(site)}><RefreshIcon /></IconButton></Tooltip>
                  <Tooltip title="Sync"><IconButton onClick={() => handleSync(site)}><SyncIcon /></IconButton></Tooltip>
                  <Tooltip title="Reindex"><IconButton onClick={() => handleReindex(site)}><SyncIcon color="secondary" /></IconButton></Tooltip>
                  <Tooltip title="Details"><IconButton onClick={() => setSelected(site)}><InfoIcon /></IconButton></Tooltip>
                  <Tooltip title="Delete"><IconButton onClick={() => setPendingDelete(site)}><DeleteIcon color="error" /></IconButton></Tooltip>
                </TableCell>
              </TableRow>
            ))}
          </TableBody>
        </Table>
      </Paper>
      <WebsiteFormDialog open={formOpen} onClose={() => setFormOpen(false)} />
      <WebsiteDetailDrawer website={selected} onClose={() => setSelected(null)} />
      <Dialog open={!!pendingDelete} onClose={() => setPendingDelete(null)}>
        <DialogTitle>Delete website</DialogTitle>
        <DialogContent>
          <DialogContentText>
            Delete "{pendingDelete?.name}"? This removes all its crawled pages and cannot be undone.
          </DialogContentText>
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setPendingDelete(null)}>Cancel</Button>
          <Button onClick={confirmDelete} color="error" variant="contained">Delete</Button>
        </DialogActions>
      </Dialog>
      <Snackbar open={!!actionError} autoHideDuration={6000} onClose={() => setActionError(null)}>
        <Alert severity="error" onClose={() => setActionError(null)} sx={{ width: '100%' }}>
          {actionError}
        </Alert>
      </Snackbar>
    </Box>
  )
}
