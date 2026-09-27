import { useCallback, useEffect, useRef, useState } from 'react'
import {
  checkUpdate,
  downloadUpdate,
  getUpdateStatus,
  restartApp,
  skipUpdate,
} from '../api'
import type { UpdateInfo, UpdateStatus } from '../api'

// Self-update UI (packaged app only; see backend/app/updater.py).
// - A newer release that the user has not skipped opens a modal.
// - "업데이트" downloads it in the background; the app keeps working and a
//   small notice shows the progress.
// - The update installs when the app closes, or right away via "지금 재시작".
// - "건너뛰기" hides this version until a newer one is released.

const FIRST_CHECK_DELAY_MS = 3_000
const RECHECK_MS = 6 * 60 * 60 * 1000
const STATUS_POLL_MS = 1_000

function UpdateNotice() {
  const [info, setInfo] = useState<UpdateInfo | null>(null)
  const [modalOpen, setModalOpen] = useState(false)
  const [status, setStatus] = useState<UpdateStatus | null>(null)
  const [busy, setBusy] = useState(false)
  const [message, setMessage] = useState<string | null>(null)
  const pollRef = useRef<number | null>(null)

  const stopPolling = () => {
    if (pollRef.current !== null) window.clearInterval(pollRef.current)
    pollRef.current = null
  }

  const pollStatus = useCallback(() => {
    stopPolling()
    pollRef.current = window.setInterval(() => {
      getUpdateStatus()
        .then((s) => {
          setStatus(s)
          if (s.state !== 'downloading') stopPolling()
        })
        .catch(() => stopPolling())
    }, STATUS_POLL_MS)
  }, [])

  useEffect(() => {
    let alive = true
    const run = async () => {
      try {
        const current = await getUpdateStatus()
        if (!alive) return
        // Downloaded earlier in this session (or still downloading).
        if (current.state === 'downloading' || current.state === 'ready') {
          setStatus(current)
          if (current.state === 'downloading') pollStatus()
          return
        }
        const found = await checkUpdate()
        if (!alive) return
        setInfo(found)
        if (found.available && !found.skipped) setModalOpen(true)
      } catch {
        // Offline or GitHub unreachable: try again at the next check.
      }
    }
    const first = window.setTimeout(run, FIRST_CHECK_DELAY_MS)
    const later = window.setInterval(run, RECHECK_MS)
    return () => {
      alive = false
      window.clearTimeout(first)
      window.clearInterval(later)
      stopPolling()
    }
  }, [pollStatus])

  const accept = async () => {
    setModalOpen(false)
    setMessage(null)
    try {
      setStatus(await downloadUpdate())
      pollStatus()
    } catch (e) {
      setMessage(e instanceof Error ? e.message : '업데이트를 받지 못했습니다.')
    }
  }

  const decline = async () => {
    setModalOpen(false)
    if (info?.latest) await skipUpdate(info.latest).catch(() => {})
  }

  const restartNow = async () => {
    setBusy(true)
    try {
      await restartApp()
    } catch (e) {
      setBusy(false)
      setMessage(e instanceof Error ? e.message : '재시작하지 못했습니다.')
    }
  }

  const version = status?.version ?? info?.latest

  return (
    <>
      {modalOpen && info?.latest && (
        <div className="modal-backdrop" role="dialog" aria-modal="true" aria-label="업데이트">
          <div className="modal">
            <p className="modal-title">새 버전이 나왔어요</p>
            <p className="modal-sub">
              v{info.current} → <strong>v{info.latest}</strong>
              {info.size ? ` · ${(info.size / 1048576).toFixed(0)}MB` : ''}
            </p>
            {info.notes && <pre className="modal-notes">{info.notes}</pre>}
            <p className="modal-hint">
              업데이트는 앱을 쓰는 동안 뒤에서 받아지고, 앱을 닫을 때 적용됩니다.
            </p>
            <div className="modal-actions">
              <button type="button" className="btn" onClick={() => void decline()}>
                이 버전 건너뛰기
              </button>
              <button type="button" className="btn btn-accent" onClick={() => void accept()}>
                업데이트
              </button>
            </div>
          </div>
        </div>
      )}

      {(status?.state === 'downloading' || status?.state === 'ready' ||
        status?.state === 'error' || message) && (
        <div className="update-toast" role="status">
          {status?.state === 'downloading' && (
            <>
              <span>v{version} 받는 중 · {Math.round(status.progress * 100)}%</span>
              <div className="progress-track update-track">
                <div className="progress-fill" style={{ width: `${status.progress * 100}%` }} />
              </div>
            </>
          )}
          {status?.state === 'ready' && (
            <>
              <span>v{version} 준비 완료 · 앱을 닫으면 적용됩니다</span>
              <button
                type="button"
                className="btn btn-sm btn-accent"
                onClick={() => void restartNow()}
                disabled={busy}
              >
                {busy ? '재시작하는 중…' : '지금 재시작'}
              </button>
            </>
          )}
          {(status?.state === 'error' || message) && (
            <>
              <span className="update-error">{message ?? status?.error}</span>
              <button type="button" className="btn btn-sm" onClick={() => void accept()}>
                다시 시도
              </button>
            </>
          )}
        </div>
      )}
    </>
  )
}

export default UpdateNotice
