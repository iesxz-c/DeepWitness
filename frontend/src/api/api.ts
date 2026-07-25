// API client — all backend calls go through here
const BASE = 'http://localhost:8000'

export type VideoUploadResponse = {
  video_id: string
  camera: string
  events_created: number
  status: string
  error?: string
}

export type VideoRecord = {
  video_id: string
  camera: string
  events_created: number
}

export type EventRecord = {
  time: string
  camera: string
  event_type: string
  description: string
  confidence: number
}

export type QueryResponse = {
  answer: string
  tools_called: string[]
  provider_used: string
}

export type ReportResponse = {
  markdown: string
  structured: unknown
  provider_used: string
}

export type EventFilters = {
  camera?: string
  event_type?: string
  start_time?: string
  end_time?: string
}

// ── Videos ────────────────────────────────────────────────────────────────────

export async function uploadVideo(
  file: File,
  camera?: string,
): Promise<VideoUploadResponse> {
  const fd = new FormData()
  fd.append('file', file)
  if (camera && camera.trim()) {
    fd.append('camera', camera.trim())
  }
  const res = await fetch(`${BASE}/videos`, { method: 'POST', body: fd })
  if (!res.ok) throw new Error(`Upload failed: ${res.status}`)
  return res.json()
}

export async function getVideos(): Promise<VideoRecord[]> {
  const res = await fetch(`${BASE}/videos`)
  if (!res.ok) throw new Error(`Fetch videos failed: ${res.status}`)
  return res.json()
}

// ── Events ────────────────────────────────────────────────────────────────────

export async function getEvents(filters: EventFilters = {}): Promise<EventRecord[]> {
  const params = new URLSearchParams()
  if (filters.camera) params.set('camera', filters.camera)
  if (filters.event_type) params.set('event_type', filters.event_type)
  if (filters.start_time) params.set('start_time', filters.start_time)
  if (filters.end_time) params.set('end_time', filters.end_time)
  const qs = params.toString()
  const res = await fetch(`${BASE}/events${qs ? `?${qs}` : ''}`)
  if (!res.ok) throw new Error(`Fetch events failed: ${res.status}`)
  return res.json()
}

// ── Query ─────────────────────────────────────────────────────────────────────

export async function postQuery(question: string): Promise<QueryResponse> {
  const res = await fetch(`${BASE}/query`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ question }),
  })
  if (!res.ok) throw new Error(`Query failed: ${res.status}`)
  return res.json()
}

// ── Report ────────────────────────────────────────────────────────────────────

export async function getReport(): Promise<ReportResponse> {
  const res = await fetch(`${BASE}/report`)
  if (!res.ok) throw new Error(`Report failed: ${res.status}`)
  return res.json()
}
