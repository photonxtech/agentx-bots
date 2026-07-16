import { useState } from 'react'
import { Box, Card, CardContent, Grid, List, ListItem, ListItemText, MenuItem, Select, Typography } from '@mui/material'
import { useAnalytics, useWebsites } from '../api/hooks'

export default function AnalyticsPage() {
  const { data: websites } = useWebsites()
  const [websiteId, setWebsiteId] = useState<number | ''>('')
  const { data } = useAnalytics(websiteId === '' ? null : websiteId)

  return (
    <Box>
      <Typography variant="h5" sx={{ mb: 2 }}>Analytics</Typography>
      <Select value={websiteId} onChange={(e) => setWebsiteId(e.target.value as number)} displayEmpty sx={{ mb: 3, minWidth: 240 }}>
        <MenuItem value="" disabled>Select a website</MenuItem>
        {websites?.map((w) => <MenuItem key={w.id} value={w.id}>{w.name}</MenuItem>)}
      </Select>

      {data && (
        <Grid container spacing={2}>
          {[
            ['Conversations', data.total_conversations],
            ['Messages', data.total_messages],
            ['Thumbs Up', data.thumbs_up],
            ['Thumbs Down', data.thumbs_down],
            ['Avg Confidence', `${data.average_confidence}%`],
            ['Fallback Rate', `${data.fallback_rate}%`],
          ].map(([label, value]) => (
            <Grid item xs={6} md={4} key={label as string}>
              <Card sx={{ borderRadius: 3 }}>
                <CardContent>
                  <Typography variant="body2" color="text.secondary">{label}</Typography>
                  <Typography variant="h5">{value}</Typography>
                </CardContent>
              </Card>
            </Grid>
          ))}
          <Grid item xs={12}>
            <Card sx={{ borderRadius: 3 }}>
              <CardContent>
                <Typography variant="subtitle1" sx={{ mb: 1 }}>Recent Questions</Typography>
                <List dense>
                  {data.recent_questions.map((q, i) => <ListItem key={i}><ListItemText primary={q} /></ListItem>)}
                </List>
              </CardContent>
            </Card>
          </Grid>
        </Grid>
      )}
    </Box>
  )
}
