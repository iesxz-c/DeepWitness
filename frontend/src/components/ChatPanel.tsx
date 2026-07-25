import { useState, useRef, useEffect, FormEvent } from 'react'
import { postQuery, type QueryResponse } from '../api/api'

type Message =
  | { role: 'user'; text: string }
  | { role: 'assistant'; data: QueryResponse }
  | { role: 'error'; text: string }

const SUGGESTIONS = [
  'What intrusion events occurred today?',
  'Which camera detected the most events?',
  'Summarise all theft incidents.',
  'Were there any weapon detections?',
]

export default function ChatPanel() {
  const [messages, setMessages] = useState<Message[]>([])
  const [input, setInput] = useState('')
  const [loading, setLoading] = useState(false)
  const bottomRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages])

  const send = async (question: string) => {
    if (!question.trim() || loading) return
    setMessages((m) => [...m, { role: 'user', text: question }])
    setInput('')
    setLoading(true)
    try {
      const data = await postQuery(question)
      setMessages((m) => [...m, { role: 'assistant', data }])
    } catch (e) {
      setMessages((m) => [...m, { role: 'error', text: String(e) }])
    } finally {
      setLoading(false)
    }
  }

  const handleSubmit = (e: FormEvent) => {
    e.preventDefault()
    send(input)
  }

  return (
    <div className="max-w-3xl flex flex-col gap-4" style={{ height: 'calc(100vh - 6rem)' }}>
      <div>
        <h1 className="text-2xl font-bold text-white">Chat / Query</h1>
        <p className="text-slate-400 mt-1 text-sm">
          Ask anything about the CCTV events in natural language.
        </p>
      </div>

      {/* Message list */}
      <div className="flex-1 card overflow-y-auto space-y-4 p-4">
        {messages.length === 0 && (
          <div className="flex flex-col items-center justify-center h-full gap-6 text-center">
            <div className="w-16 h-16 rounded-2xl bg-indigo-600/20 border border-indigo-600/40 flex items-center justify-center text-3xl">
              💬
            </div>
            <div>
              <p className="text-slate-300 font-medium">Ask about your CCTV data</p>
              <p className="text-slate-500 text-sm mt-1">Try one of the suggestions below</p>
            </div>
            <div className="flex flex-wrap gap-2 justify-center">
              {SUGGESTIONS.map((s) => (
                <button
                  key={s}
                  className="btn-ghost text-xs px-3 py-1.5"
                  onClick={() => send(s)}
                >
                  {s}
                </button>
              ))}
            </div>
          </div>
        )}

        {messages.map((msg, i) => {
          if (msg.role === 'user') {
            return (
              <div key={i} className="flex justify-end">
                <div className="max-w-[80%] bg-indigo-600 text-white px-4 py-2.5 rounded-2xl rounded-tr-sm text-sm leading-relaxed shadow-lg shadow-indigo-900/30">
                  {msg.text}
                </div>
              </div>
            )
          }

          if (msg.role === 'error') {
            return (
              <div key={i} className="flex justify-start">
                <div className="max-w-[80%] bg-red-950/50 border border-red-700/50 text-red-300 px-4 py-2.5 rounded-2xl rounded-tl-sm text-sm">
                  ⚠ {msg.text}
                </div>
              </div>
            )
          }

          // assistant
          return (
            <div key={i} className="flex justify-start gap-2 items-start">
              <div className="shrink-0 w-7 h-7 bg-indigo-700/30 border border-indigo-700/40 rounded-lg flex items-center justify-center text-sm">
                🤖
              </div>
              <div className="max-w-[85%] space-y-2">
                <div className="bg-slate-800 text-slate-200 px-4 py-3 rounded-2xl rounded-tl-sm text-sm leading-relaxed">
                  {msg.data.answer}
                </div>
                {/* Tool chips */}
                {msg.data.tools_called?.length > 0 && (
                  <div className="flex flex-wrap gap-1.5 px-1">
                    <span className="text-[10px] text-slate-600 uppercase tracking-widest self-center">
                      Tools:
                    </span>
                    {msg.data.tools_called.map((t) => (
                      <span
                        key={t}
                        className="text-[10px] bg-slate-800 border border-slate-700 text-slate-400 px-2 py-0.5 rounded-full font-mono"
                      >
                        {t}
                      </span>
                    ))}
                  </div>
                )}
                {msg.data.provider_used && (
                  <p className="text-[10px] text-slate-600 px-1">
                    via {msg.data.provider_used}
                  </p>
                )}
              </div>
            </div>
          )
        })}

        {/* Loading bubble */}
        {loading && (
          <div className="flex justify-start gap-2 items-start">
            <div className="shrink-0 w-7 h-7 bg-indigo-700/30 border border-indigo-700/40 rounded-lg flex items-center justify-center text-sm">
              🤖
            </div>
            <div className="bg-slate-800 px-4 py-3 rounded-2xl rounded-tl-sm">
              <div className="flex gap-1 items-center">
                {[0, 1, 2].map((d) => (
                  <span
                    key={d}
                    className="w-1.5 h-1.5 bg-slate-500 rounded-full animate-bounce"
                    style={{ animationDelay: `${d * 0.15}s` }}
                  />
                ))}
              </div>
            </div>
          </div>
        )}

        <div ref={bottomRef} />
      </div>

      {/* Input */}
      <form onSubmit={handleSubmit} className="flex gap-2">
        <input
          id="chat-input"
          type="text"
          className="input flex-1"
          placeholder="Ask about your CCTV events…"
          value={input}
          onChange={(e) => setInput(e.target.value)}
          disabled={loading}
          autoComplete="off"
        />
        <button
          id="chat-send-btn"
          type="submit"
          className="btn-primary px-5"
          disabled={!input.trim() || loading}
        >
          {loading ? <span className="spinner" /> : '↑'}
        </button>
      </form>
    </div>
  )
}
