export interface Website {
  id: number
  url: string
  name: string
  logo_url: string | null
  status: 'active' | 'crawling' | 'error'
  sitemap_url: string | null
  crawl_depth_limit: number
  max_pages: number
  created_at: string
  updated_at: string
  last_synced_at: string | null
}

export interface Page {
  id: number
  website_id: number
  url: string
  title: string | null
  status: 'indexed' | 'failed' | 'skipped' | 'deleted'
  content_hash: string | null
  last_crawled_at: string | null
  indexed_at: string | null
}

export interface CrawlJob {
  id: number
  website_id: number
  status: 'running' | 'completed' | 'failed'
  pages_found: number
  pages_indexed: number
  pages_failed: number
  started_at: string
  finished_at: string | null
  error_log: string | null
}

export interface MessageSource {
  page_id: number
  url: string
  title: string
}

export interface ChatMessage {
  id: number
  role: 'user' | 'assistant'
  content: string
  sources: MessageSource[] | null
  confidence: number | null
  feedback: 'up' | 'down' | null
  created_at: string
}

export interface Conversation {
  id: number
  website_id: number
  session_id: string
  title: string
  created_at: string
}

export interface ConversationDetail extends Conversation {
  messages: ChatMessage[]
}

export interface AnalyticsSummary {
  total_conversations: number
  total_messages: number
  thumbs_up: number
  thumbs_down: number
  average_confidence: number
  fallback_rate: number
  recent_questions: string[]
}
