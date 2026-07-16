import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import { apiClient } from './client'
import type { AnalyticsSummary, CrawlJob, Page, Website } from './types'

export function useWebsites() {
  return useQuery({
    queryKey: ['websites'],
    queryFn: async () => (await apiClient.get<Website[]>('/websites')).data,
    // Crawl/sync/reindex run in a background task after the triggering mutation's
    // onSuccess already fired, so status transitions (crawling -> active/error) and
    // last_synced_at wouldn't otherwise show up until something else refetches.
    refetchInterval: 5000,
  })
}

export function useCreateWebsite() {
  const qc = useQueryClient()
  return useMutation({
    mutationFn: (payload: { url: string; name: string; logo_url?: string }) =>
      apiClient.post<Website>('/websites', payload),
    onSuccess: () => qc.invalidateQueries({ queryKey: ['websites'] }),
  })
}

export function useUpdateWebsite() {
  const qc = useQueryClient()
  return useMutation({
    mutationFn: ({ id, ...payload }: { id: number; name?: string; logo_url?: string }) =>
      apiClient.patch<Website>(`/websites/${id}`, payload),
    onSuccess: () => qc.invalidateQueries({ queryKey: ['websites'] }),
  })
}

export function useDeleteWebsite() {
  const qc = useQueryClient()
  return useMutation({
    mutationFn: (id: number) => apiClient.delete(`/website/${id}`),
    onSuccess: () => qc.invalidateQueries({ queryKey: ['websites'] }),
  })
}

function useTriggerAction(path: (id: number) => string) {
  const qc = useQueryClient()
  return useMutation({
    mutationFn: (id: number) => apiClient.post(path(id)),
    onSuccess: () => qc.invalidateQueries({ queryKey: ['websites'] }),
  })
}

export const useCrawlWebsite = () => useTriggerAction((id) => `/crawl/${id}`)
export const useSyncWebsite = () => useTriggerAction((id) => `/sync/${id}`)
export const useReindexWebsite = () => useTriggerAction((id) => `/reindex/${id}`)

export function usePages(websiteId: number | null) {
  return useQuery({
    queryKey: ['pages', websiteId],
    queryFn: async () => (await apiClient.get<Page[]>(`/pages/${websiteId}`)).data,
    enabled: websiteId !== null,
  })
}

export function useCrawlJobs(websiteId: number | null) {
  return useQuery({
    queryKey: ['crawl-jobs', websiteId],
    queryFn: async () => (await apiClient.get<CrawlJob[]>(`/websites/${websiteId}/crawl-jobs`)).data,
    enabled: websiteId !== null,
    refetchInterval: 5000,
  })
}

export function useAnalytics(websiteId: number | null) {
  return useQuery({
    queryKey: ['analytics', websiteId],
    queryFn: async () => (await apiClient.get<AnalyticsSummary>(`/analytics/${websiteId}`)).data,
    enabled: websiteId !== null,
  })
}
