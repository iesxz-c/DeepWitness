import { BrowserRouter, Routes, Route } from 'react-router-dom'
import Layout from './components/Layout'
import VideoUpload from './components/VideoUpload'
import VideosList from './components/VideosList'
import EventTimeline from './components/EventTimeline'
import ChatPanel from './components/ChatPanel'
import ReportViewer from './components/ReportViewer'

export default function App() {
  return (
    <BrowserRouter>
      <Layout>
        <Routes>
          <Route path="/" element={<VideoUpload />} />
          <Route path="/videos" element={<VideosList />} />
          <Route path="/timeline" element={<EventTimeline />} />
          <Route path="/chat" element={<ChatPanel />} />
          <Route path="/report" element={<ReportViewer />} />
        </Routes>
      </Layout>
    </BrowserRouter>
  )
}
