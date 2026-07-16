// Design tokens: Photonx -> photon -> warm amber as the one accent, on a clean
// bubble-based chat layout. These are the only hex values used across the widget/theme.

export const tokens = {
  dark: {
    void: '#111318', // page/widget background
    panel: '#1A1D24', // header / input surface
    bubble: '#23262F', // assistant bubble fill
    line: '#2A2D36', // hairlines, dividers
    paper: '#F2EFEA', // primary text
    ash: '#8D93A1', // secondary text, timestamps, placeholders
    filament: '#E8A33D', // the one accent: warm amber, the "photon"
    filamentDim: '#4A3A20',
    ember: '#E5695F', // negative state, used sparingly
  },
  light: {
    void: '#F4F2ED',
    panel: '#FFFFFF',
    bubble: '#F0EFEC', // assistant bubble fill
    line: '#E7E2D8',
    paper: '#1A1720',
    ash: '#6B6577',
    filament: '#C97A1A', // burnt amber — enough contrast on paper
    filamentDim: '#E4C9A0',
    ember: '#C4453B',
  },
  font: {
    display: '"Space Grotesk", "Segoe UI", sans-serif',
    body: '"Inter", "Segoe UI", sans-serif',
    mono: '"IBM Plex Mono", ui-monospace, monospace',
  },
} as const

export type SignalPalette = typeof tokens.dark
