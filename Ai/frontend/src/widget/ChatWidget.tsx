import { useState } from 'react'
import { AnimatePresence } from 'framer-motion'
import FloatingButton from './FloatingButton'
import ChatWindow from './ChatWindow'
import type { Website } from '../api/types'

export default function ChatWidget({ website }: { website: Website }) {
  const [open, setOpen] = useState(false)

  return (
    <>
      <AnimatePresence>
        {open && <ChatWindow key="chat-window" website={website} onClose={() => setOpen(false)} />}
      </AnimatePresence>
      <FloatingButton open={open} onClick={() => setOpen((v) => !v)} />
    </>
  )
}
