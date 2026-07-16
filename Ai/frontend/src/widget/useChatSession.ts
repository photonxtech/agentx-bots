import { useCallback, useRef, useState } from 'react'
import { apiClient } from '../api/client'
import type { ChatMessage, MessageSource } from '../api/types'

function getSessionId(websiteId: number): string {
  const key = `chat-session-${websiteId}`
  let id = localStorage.getItem(key)
  if (!id) {
    id = crypto.randomUUID()
    localStorage.setItem(key, id)
  }
  return id
}

export interface StreamingMessage {
  id: number | 'streaming'
  role: 'user' | 'assistant'
  content: string
  sources?: MessageSource[]
  confidence?: number
  feedback?: 'up' | 'down' | null
}

export function useChatSession(websiteId: number) {
  const sessionId = useRef(getSessionId(websiteId)).current
  const [conversationId, setConversationId] = useState<number | null>(null)
  const [messages, setMessages] = useState<StreamingMessage[]>([])
  const [isStreaming, setIsStreaming] = useState(false)
  const abortRef = useRef<AbortController | null>(null)

  const sendMessage = useCallback(
    async (question: string, options?: { regenerate?: boolean }) => {
      setMessages((prev) => [...prev, { id: Date.now(), role: 'user', content: question }])
      setMessages((prev) => [...prev, { id: 'streaming', role: 'assistant', content: '' }])
      setIsStreaming(true)

      const controller = new AbortController()
      abortRef.current = controller

      const baseURL = apiClient.defaults.baseURL ?? ''
      let response: Response
      try {
        response = await fetch(`${baseURL}/chat`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            website_id: websiteId, question, session_id: sessionId, conversation_id: conversationId,
            regenerate: options?.regenerate ?? false,
          }),
          signal: controller.signal,
        })
      } catch {
        setIsStreaming(false)
        return
      }

      const reader = response.body?.getReader()
      const decoder = new TextDecoder()
      let buffer = ''
      if (!reader) {
        setIsStreaming(false)
        return
      }

      try {
        while (true) {
          const { done, value } = await reader.read()
          if (done) break
          buffer += decoder.decode(value, { stream: true })

          const blocks = buffer.split('\n\n')
          buffer = blocks.pop() ?? ''

          for (const block of blocks) {
            if (!block.startsWith('data: ')) continue
            const event = JSON.parse(block.slice(6))

            if (event.type === 'delta') {
              setMessages((prev) =>
                prev.map((m) => (m.id === 'streaming' ? { ...m, content: m.content + event.content } : m)),
              )
            } else if (event.type === 'done') {
              setConversationId(event.conversation_id)
              setMessages((prev) =>
                prev.map((m) =>
                  m.id === 'streaming'
                    ? { ...m, id: event.message_id, sources: event.sources, confidence: event.confidence, feedback: null }
                    : m,
                ),
              )
            }
          }
        }
      } catch {
        // aborted by the user via stopStreaming() — the partial reply already
        // typed into the "streaming" message is left in place, just unlocked.
      } finally {
        setIsStreaming(false)
        abortRef.current = null
      }
    },
    [websiteId, sessionId, conversationId],
  )

  const stopStreaming = useCallback(() => {
    abortRef.current?.abort()
  }, [])

  const clearChat = useCallback(() => {
    stopStreaming()
    setMessages([])
    setConversationId(null)
  }, [stopStreaming])

  const loadConversation = useCallback(async (id: number) => {
    const { data } = await apiClient.get(`/conversations/${id}`)
    setConversationId(id)
    setMessages(data.messages.map((m: ChatMessage) => ({ ...m })))
  }, [])

  return { sessionId, conversationId, messages, isStreaming, sendMessage, stopStreaming, clearChat, loadConversation }
}
