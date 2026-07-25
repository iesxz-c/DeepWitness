import { useEffect, useState } from 'react'
import { getEvents, type EventRecord, type EventFilters } from '../api/api'

const EVENT_COLORS: Record<string, { bg: string; text: string; dot: string }> = {
  intrusion:       { bg: 'bg-red-900/40',    text: 'text-red-300',    dot: 'bg-red-500' },
  theft:           { bg: 'bg-orange-900/40', text: 'text-orange-300', dot: 'bg-orange-500' },
  loitering:       { bg: 'bg-yellow-900/40', text: 'text-yellow-300', dot: 'bg-yellow-500' },
  vandalism:       { bg: 'bg-purple-900/40', text: 'text-purple-300', dot: 'bg-purple-500' },
  weapon_detected: { bg: 'bg-rose-900/40',   text: 'text-rose-300',   dot: 'bg-rose-500' },
}

const DEFAULT_COLOR = { bg: 'bg-slate-800/50', text: 'text-slate-400', dot: 'bg-slate-500' }

function getColor(eventType: string) {
  return EVENT_COLORS[eventType] ?? DEFAULT_COLOR
}

function ConfidencePill({ value }: { value: number }) {
  const pct = Math.round(value * 100)
  const color =
    pct >= 85 ? 'bg-green-600' : pct >= 65 ? 'bg-yellow-600' : 'bg-red-600'
  return (
    <div className="flex items-center gap-1.5">
      <div className="w-16 h-1.5 bg-slate-700 rounded-full overflow-hidden">
        <div className={`h-full ${color} rounded-full`} style={{ width: `${pct}%` }} />
      </div>
      <span className="text-xs text-slate-400 tabular-nums">{pct}%</span>
    </div>
  )
}

export default function EventTimeline() {
  const [events, setEvents] = useState<EventRecord[]>([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [filters, setFilters] = useState<EventFilters>({})

  const fetchEvents = async () => {
    setLoading(true)
    setError(null)
    try {
      const data = await getEvents(filters)
      const sorted = [...data].sort((a, b) => a.time.localeCompare(b.time))
      setEvents(sorted)
    } catch (e) {
      setError(String(e))
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => { fetchEvents() }, [])

  const cameras = [...new Set(events.map((e) => e.camera))]
  const eventTypes = [...new Set(events.map((e) => e.event_type))]

  return (
    <div className="max-w-4xl space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold text-white">Event Timeline</h1>
          <p className="text-slate-400 mt-1 text-sm">Chronological view of all detected events.</p>
        </div>
        <button id="refresh-events-btn" onClick={fetchEvents} className="btn-ghost text-sm">
          🔄 Refresh
        </button>
      </div>

      {/* Filters */}
      <div className="card flex flex-wrap gap-3 py-4">
        <select
          className="input w-auto"
          value={filters.camera ?? ''}
          onChange={(e) => setFilters((f) => ({ ...f, camera: e.target.value || undefined }))}
        >
          <option value="">All cameras</option>
          {cameras.map((c) => <option key={c} value={c}>{c}</option>)}
        </select>
        <select
          className="input w-auto"
          value={filters.event_type ?? ''}
          onChange={(e) => setFilters((f) => ({ ...f, event_type: e.target.value || undefined }))}
        >
          <option value="">All event types</option>
          {eventTypes.map((t) => <option key={t} value={t}>{t}</option>)}
        </select>
        <button
          id="apply-filters-btn"
          className="btn-primary text-sm"
          onClick={fetchEvents}
        >
          Apply
        </button>
        <button
          className="btn-ghost text-sm"
          onClick={() => { setFilters({}); setTimeout(fetchEvents, 0) }}
        >
          Clear
        </button>
      </div>

      {/* Timeline */}
      {loading ? (
        <div className="flex justify-center py-20">
          <span className="spinner border-indigo-500 border-t-transparent" />
        </div>
      ) : error ? (
        <div className="card text-red-400 text-sm">{error}</div>
      ) : events.length === 0 ? (
        <div className="card flex flex-col items-center gap-3 py-16 text-center">
          <span className="text-4xl">📭</span>
          <p className="text-slate-400">No events match the current filters.</p>
        </div>
      ) : (
        <div className="space-y-3">
          {events.map((ev, i) => {
            const color = getColor(ev.event_type)
            return (
              <div
                key={`${ev.time}-${i}`}
                className={`card border-0 ${color.bg} flex gap-4 items-start`}
              >
                {/* Time */}
                <div className="shrink-0 pt-1">
                  <span className="font-mono text-xs text-slate-400 tabular-nums">{ev.time}</span>
                </div>

                {/* Dot */}
                <div className="shrink-0 mt-1.5">
                  <span className={`block w-2.5 h-2.5 rounded-full ${color.dot} shadow-lg`} />
                </div>

                {/* Content */}
                <div className="flex-1 min-w-0 space-y-1.5">
                  <div className="flex items-center gap-2 flex-wrap">
                    <span className={`badge ${color.bg} ${color.text} border border-current/20`}>
                      {ev.event_type}
                    </span>
                    <span className="text-slate-500 text-xs">📷 {ev.camera}</span>
                  </div>
                  <p className="text-slate-200 text-sm leading-relaxed">{ev.description}</p>
                  <ConfidencePill value={ev.confidence} />
                </div>
              </div>
            )
          })}
        </div>
      )}

      {!loading && events.length > 0 && (
        <p className="text-xs text-slate-600 text-right">{events.length} event{events.length !== 1 ? 's' : ''}</p>
      )}
    </div>
  )
}
