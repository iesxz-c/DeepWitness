import { useState, useRef, DragEvent, ChangeEvent } from 'react'
import { uploadVideo, type VideoUploadResponse } from '../api/api'

export default function VideoUpload() {
  const [camera, setCamera] = useState('')
  const [file, setFile] = useState<File | null>(null)
  const [dragging, setDragging] = useState(false)
  const [loading, setLoading] = useState(false)
  const [result, setResult] = useState<VideoUploadResponse | null>(null)
  const [error, setError] = useState<string | null>(null)
  const inputRef = useRef<HTMLInputElement>(null)

  const handleDrop = (e: DragEvent) => {
    e.preventDefault()
    setDragging(false)
    const dropped = e.dataTransfer.files[0]
    if (dropped) setFile(dropped)
  }

  const handleFileChange = (e: ChangeEvent<HTMLInputElement>) => {
    if (e.target.files?.[0]) setFile(e.target.files[0])
  }

  const handleUpload = async () => {
    if (!file) return
    setLoading(true)
    setError(null)
    setResult(null)
    try {
      const res = await uploadVideo(file, camera)
      if (res.error) {
        setError(res.error)
      } else {
        setResult(res)
        setFile(null)
        setCamera('')
        if (inputRef.current) inputRef.current.value = ''
      }
    } catch (err) {
      setError(String(err))
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="max-w-2xl space-y-6">
      <div>
        <h1 className="text-2xl font-bold text-white">Upload Video</h1>
        <p className="text-slate-400 mt-1 text-sm">
          Submit a CCTV clip for AI-powered event detection.
        </p>
      </div>

      <div className="card space-y-5">
        {/* Drop zone */}
        <div
          className={`drop-zone ${dragging ? 'dragging' : ''}`}
          onDragOver={(e) => { e.preventDefault(); setDragging(true) }}
          onDragLeave={() => setDragging(false)}
          onDrop={handleDrop}
          onClick={() => inputRef.current?.click()}
        >
          <input
            ref={inputRef}
            type="file"
            accept="video/*"
            className="hidden"
            onChange={handleFileChange}
          />
          <div className="flex flex-col items-center gap-3 pointer-events-none">
            <div className="w-14 h-14 rounded-2xl bg-indigo-600/20 border border-indigo-600/40 flex items-center justify-center text-2xl">
              🎥
            </div>
            {file ? (
              <div className="text-center">
                <p className="text-white font-medium">{file.name}</p>
                <p className="text-slate-400 text-sm mt-0.5">
                  {(file.size / 1024 / 1024).toFixed(2)} MB
                </p>
              </div>
            ) : (
              <div className="text-center">
                <p className="text-slate-300 font-medium">Drop video here or click to browse</p>
                <p className="text-slate-500 text-xs mt-1">Supports MP4, AVI, MOV, MKV</p>
              </div>
            )}
          </div>
        </div>

        {/* Camera name */}
        <div>
          <label className="block text-sm font-medium text-slate-400 mb-1.5">
            Camera Name <span className="text-slate-600">(optional)</span>
          </label>
          <input
            type="text"
            className="input"
            placeholder="e.g. cam-entrance"
            value={camera}
            onChange={(e) => setCamera(e.target.value)}
          />
        </div>

        {/* Submit */}
        <button
          id="upload-btn"
          className="btn-primary w-full justify-center py-3"
          disabled={!file || loading}
          onClick={handleUpload}
        >
          {loading ? (
            <>
              <span className="spinner" />
              Processing…
            </>
          ) : (
            <>
              <span>⬆</span>
              Upload & Analyse
            </>
          )}
        </button>

        {/* Error */}
        {error && (
          <div className="flex items-start gap-2 bg-red-950/50 border border-red-700/50 rounded-xl p-4">
            <span className="text-red-400 text-lg leading-none mt-0.5">⚠</span>
            <p className="text-red-300 text-sm">{error}</p>
          </div>
        )}
      </div>

      {/* Success card */}
      {result && (
        <div
          id="upload-success"
          className="card border-indigo-700/50 bg-indigo-950/30 space-y-4 animate-pulse-once"
        >
          <div className="flex items-center gap-3">
            <div className="w-10 h-10 rounded-xl bg-green-600/20 border border-green-600/30 flex items-center justify-center text-xl">
              ✅
            </div>
            <div>
              <p className="font-semibold text-white">Video processed!</p>
              <p className="text-slate-400 text-xs">Status: {result.status}</p>
            </div>
          </div>
          <div className="grid grid-cols-3 gap-3">
            {[
              { label: 'Video ID', value: result.video_id },
              { label: 'Camera', value: result.camera },
              { label: 'Events Created', value: String(result.events_created) },
            ].map(({ label, value }) => (
              <div key={label} className="bg-slate-800/60 rounded-xl p-3">
                <p className="text-[10px] uppercase tracking-widest text-slate-500 mb-1">{label}</p>
                <p className="text-white font-semibold text-sm truncate">{value}</p>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  )
}
