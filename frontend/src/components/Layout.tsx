import { NavLink } from 'react-router-dom'

const links = [
  { to: '/', label: 'Upload Video', icon: '⬆' },
  { to: '/videos', label: 'Videos', icon: '🎬' },
  { to: '/timeline', label: 'Event Timeline', icon: '📋' },
  { to: '/chat', label: 'Chat / Query', icon: '💬' },
  { to: '/report', label: 'Report', icon: '📄' },
]

export default function Layout({ children }: { children: React.ReactNode }) {
  return (
    <div className="flex min-h-screen">
      {/* Sidebar */}
      <aside className="fixed top-0 left-0 h-screen w-[220px] bg-slate-900 border-r border-slate-800 flex flex-col z-20">
        {/* Logo */}
        <div className="px-6 py-5 border-b border-slate-800">
          <div className="flex items-center gap-3">
            <div className="w-8 h-8 bg-indigo-600 rounded-lg flex items-center justify-center text-white text-base shadow-lg shadow-indigo-900/50">
              🔒
            </div>
            <div>
              <p className="text-sm font-bold text-white leading-tight">CCTV Intel</p>
              <p className="text-[10px] text-slate-500 uppercase tracking-widest">Dashboard</p>
            </div>
          </div>
        </div>

        {/* Nav */}
        <nav className="flex-1 px-3 py-4 space-y-1 overflow-y-auto">
          {links.map(({ to, label, icon }) => (
            <NavLink
              key={to}
              to={to}
              end={to === '/'}
              className={({ isActive }) =>
                `nav-item ${isActive ? 'active' : ''}`
              }
            >
              <span className="text-base">{icon}</span>
              <span>{label}</span>
            </NavLink>
          ))}
        </nav>

        {/* Footer */}
        <div className="px-4 py-3 border-t border-slate-800">
          <p className="text-[10px] text-slate-600 uppercase tracking-widest">localhost:8000</p>
        </div>
      </aside>

      {/* Main content */}
      <main className="ml-[220px] flex-1 h-screen overflow-y-auto bg-slate-950 p-8">
        {children}
      </main>
    </div>
  )
}
