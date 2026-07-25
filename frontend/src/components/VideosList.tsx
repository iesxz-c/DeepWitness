import { useEffect, useState } from 'react'
import { getVideos, type VideoRecord } from '../api/api'

export default function VideosList() {
  const [videos, setVideos] = useState<VideoRecord[]>([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  const fetchVideos = async () => {
    setLoading(true)
    setError(null)
    try {
      setVideos(await getVideos())
    } catch (e) {
      setError(String(e))
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => { fetchVideos() }, [])

  return (
    <div className="max-w-4xl space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold text-white">Uploaded Videos</h1>
          <p className="text-slate-400 mt-1 text-sm">All processed video clips and event counts.</p>
        </div>
        <button
          id="refresh-videos-btn"
          onClick={fetchVideos}
          className="btn-ghost text-sm"
        >
          🔄 Refresh
        </button>
      </div>

      <div className="card p-0 overflow-hidden">
        {loading ? (
          <div className="flex items-center justify-center py-20">
            <span className="spinner border-indigo-500 border-t-transparent" />
          </div>
        ) : error ? (
          <div className="p-6 text-red-400 text-sm">{error}</div>
        ) : videos.length === 0 ? (
          <div className="flex flex-col items-center gap-3 py-20 text-center">
            <span className="text-4xl">📭</span>
            <p className="text-slate-400">No videos uploaded yet.</p>
            <a href="/" className="text-indigo-400 text-sm hover:underline">Upload your first video →</a>
          </div>
        ) : (
          <table className="w-full">
            <thead className="border-b border-slate-800 bg-slate-900/50">
              <tr>
                <th className="table-th">Video ID</th>
                <th className="table-th">Camera</th>
                <th className="table-th">Events Created</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-800">
              {videos.map((v) => (
                <tr key={v.video_id} className="hover:bg-slate-800/30 transition-colors">
                  <td className="table-td">
                    <span className="font-mono text-indigo-400 text-xs bg-indigo-950/40 px-2 py-0.5 rounded-lg">
                      {v.video_id}
                    </span>
                  </td>
                  <td className="table-td">
                    <span className="flex items-center gap-1.5">
                      <span className="w-2 h-2 rounded-full bg-green-500" />
                      {v.camera}
                    </span>
                  </td>
                  <td className="table-td">
                    <span className={`badge ${v.events_created > 0 ? 'bg-indigo-900/50 text-indigo-300' : 'bg-slate-800 text-slate-400'}`}>
                      {v.events_created} event{v.events_created !== 1 ? 's' : ''}
                    </span>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </div>

      {!loading && videos.length > 0 && (
        <p className="text-xs text-slate-600 text-right">{videos.length} video{videos.length !== 1 ? 's' : ''} total</p>
      )}
    </div>
  )
}
