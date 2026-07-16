import { createTheme, type Theme } from '@mui/material/styles'
import { tokens } from './tokens'

export function buildTheme(mode: 'light' | 'dark'): Theme {
  const t = mode === 'dark' ? tokens.dark : tokens.light

  return createTheme({
    palette: {
      mode,
      primary: { main: t.filament, contrastText: mode === 'dark' ? '#0A0A0C' : '#FFFFFF' },
      secondary: { main: t.ash },
      error: { main: t.ember },
      background: { default: t.void, paper: t.panel },
      text: { primary: t.paper, secondary: t.ash },
      divider: t.line,
    },
    shape: { borderRadius: 14 },
    typography: {
      fontFamily: tokens.font.body,
      h5: { fontFamily: tokens.font.display, fontWeight: 600 },
      h6: { fontFamily: tokens.font.display, fontWeight: 600 },
      subtitle1: { fontFamily: tokens.font.display, fontWeight: 600 },
      subtitle2: { fontFamily: tokens.font.display, fontWeight: 600 },
    },
  })
}
