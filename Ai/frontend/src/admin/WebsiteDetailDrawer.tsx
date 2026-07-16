import { Box, Chip, Drawer, List, ListItem, ListItemText, Typography } from '@mui/material'
import { usePages, useCrawlJobs } from '../api/hooks'
import type { Website } from '../api/types'

export default function WebsiteDetailDrawer({ website, onClose }: { website: Website | null; onClose: () => void }) {
  const { data: pages } = usePages(website?.id ?? null)
  const { data: jobs } = useCrawlJobs(website?.id ?? null)

  return (
    <Drawer anchor="right" open={website !== null} onClose={onClose}>
      <Box sx={{ width: 420, p: 3 }}>
        <Typography variant="h6">{website?.name}</Typography>
        <Typography variant="subtitle2" sx={{ mt: 2 }}>Crawl Jobs</Typography>
        <List dense>
          {jobs?.map((job) => (
            <ListItem key={job.id}>
              <ListItemText
                primary={`${job.status} — found ${job.pages_found}, indexed ${job.pages_indexed}, failed ${job.pages_failed}`}
                secondary={new Date(job.started_at).toLocaleString()}
              />
            </ListItem>
          ))}
        </List>
        <Typography variant="subtitle2" sx={{ mt: 2 }}>Indexed Pages ({pages?.length ?? 0})</Typography>
        <List dense>
          {pages?.map((page) => (
            <ListItem key={page.id}>
              <ListItemText primary={page.title || page.url} secondary={page.url} />
              <Chip size="small" label={page.status} color={page.status === 'indexed' ? 'success' : page.status === 'failed' ? 'error' : 'default'} />
            </ListItem>
          ))}
        </List>
      </Box>
    </Drawer>
  )
}
