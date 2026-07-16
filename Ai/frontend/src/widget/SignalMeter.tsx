import { Box, Stack, Typography } from '@mui/material'

// Signature element: confidence reads as signal strength, not a percentage chip —
// a deliberate nod to Photonx / photon / transmitted light.
export default function SignalMeter({ confidence }: { confidence: number }) {
  const bars = [1, 2, 3, 4, 5]
  const filled = Math.max(0, Math.min(5, Math.ceil((confidence / 100) * 5)))

  return (
    <Stack direction="row" alignItems="flex-end" spacing={0.5}>
      <Stack direction="row" alignItems="flex-end" spacing="2px" sx={{ height: 12 }}>
        {bars.map((bar) => (
          <Box
            key={bar}
            sx={{
              width: 3,
              height: `${bar * 20}%`,
              borderRadius: '1px',
              bgcolor: bar <= filled ? 'primary.main' : 'divider',
              transition: 'background-color 0.2s ease',
            }}
          />
        ))}
      </Stack>
      <Typography
        variant="caption"
        sx={{ fontFamily: '"IBM Plex Mono", monospace', color: 'text.secondary', fontSize: 11, lineHeight: 1 }}
      >
        {Math.round(confidence)}%
      </Typography>
    </Stack>
  )
}
