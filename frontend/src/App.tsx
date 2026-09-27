import { useCallback, useState } from 'react'
import type { DocumentInfo } from './api'
import UpdateNotice from './components/UpdateNotice'
import BatchPage from './pages/BatchPage'
import ViewerPage from './pages/ViewerPage'

// Two-state machine (no router): the batch board, or one finished paper in
// the viewer. The board keeps its jobs in localStorage (ids only), so it is
// restored when the viewer is closed.
type Phase = { name: 'board' } | { name: 'viewer'; doc: DocumentInfo }

function App() {
  const [phase, setPhase] = useState<Phase>({ name: 'board' })

  const handleOpen = useCallback((doc: DocumentInfo) => {
    setPhase({ name: 'viewer', doc })
  }, [])

  const handleBack = useCallback(() => {
    setPhase({ name: 'board' })
  }, [])

  // The update notice stays mounted across both pages, so a background
  // download keeps showing its progress while the user reads a paper.
  return (
    <>
      {phase.name === 'board' ? (
        <BatchPage onOpen={handleOpen} />
      ) : (
        <ViewerPage doc={phase.doc} onReset={handleBack} />
      )}
      <UpdateNotice />
    </>
  )
}

export default App
