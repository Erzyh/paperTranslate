import { useCallback, useEffect, useRef, useState } from 'react'
import { requestExplain, segmentPngUrl } from '../api'
import type { ExplainKind } from '../api'

// Right-hand drawer that streams an AI explanation for a text selection or a
// formula segment. The parent remounts this component (via a React key) for
// every new request, so all state below always belongs to the current one.

type Status = 'streaming' | 'done' | 'error'

interface Props {
  docId: string
  /** Source text sent to the explain endpoint (shown verbatim on top). */
  text: string
  kind: ExplainKind
  /**
   * Segment id for formula requests: the backend re-extracts the model input
   * from the segment's PDF region and the header shows its cropped image.
   */
  segmentId?: string
  /** Reports streaming state so the parent can block duplicate requests. */
  onBusyChange: (busy: boolean) => void
  onClose: () => void
}

function ExplainPanel({ docId, text, kind, segmentId, onBusyChange, onClose }: Props) {
  const bodyRef = useRef<HTMLDivElement>(null)
  const controllerRef = useRef<AbortController | null>(null)
  // Set when the user pressed "중단": the abort is intentional, not an error.
  const stoppedRef = useRef(false)
  const [answer, setAnswer] = useState('')
  const [status, setStatus] = useState<Status>('streaming')
  const [errorMsg, setErrorMsg] = useState('')

  useEffect(() => {
    let disposed = false
    const controller = new AbortController()
    controllerRef.current = controller
    onBusyChange(true)

    const run = async () => {
      const res = await requestExplain(docId, text, kind, controller.signal, segmentId)
      if (disposed) return
      if (res.status === 503) {
        setStatus('error')
        setErrorMsg('AI 서버에 연결할 수 없습니다. 잠시 후 다시 시도해주세요.')
        return
      }
      if (!res.ok || res.body === null) {
        setStatus('error')
        setErrorMsg('설명 요청에 실패했습니다.')
        return
      }
      const reader = res.body.getReader()
      const decoder = new TextDecoder()
      for (;;) {
        const { value, done } = await reader.read()
        if (disposed) return
        if (done) break
        const chunk = decoder.decode(value, { stream: true })
        if (chunk !== '') setAnswer((prev) => prev + chunk)
      }
      const tail = decoder.decode()
      if (tail !== '') setAnswer((prev) => prev + tail)
      setStatus('done')
    }

    run()
      .catch(() => {
        if (disposed) return
        if (stoppedRef.current) {
          // Intentional stop: keep whatever partial answer already streamed.
          setStatus('done')
          return
        }
        setStatus('error')
        setErrorMsg('설명을 가져오는 중 오류가 발생했습니다.')
      })
      .finally(() => {
        if (!disposed) onBusyChange(false)
      })

    return () => {
      disposed = true
      controller.abort()
      onBusyChange(false)
    }
  }, [docId, text, kind, segmentId, onBusyChange])

  // Keep the newest streamed text in view while the answer grows.
  useEffect(() => {
    if (status !== 'streaming') return
    const el = bodyRef.current
    if (el) el.scrollTop = el.scrollHeight
  }, [answer, status])

  const handleStop = useCallback(() => {
    stoppedRef.current = true
    controllerRef.current?.abort()
  }, [])

  return (
    <aside className="explain-drawer">
      <div className="explain-head">
        <span className="explain-title">AI 설명</span>
        {status === 'streaming' ? (
          <button type="button" className="btn btn-sm" onClick={handleStop}>
            중단
          </button>
        ) : (
          <button type="button" className="btn btn-sm" onClick={onClose}>
            닫기
          </button>
        )}
      </div>
      <div className="explain-body" ref={bodyRef}>
        <p className="explain-query-label">{kind === 'formula' ? '수식' : '선택한 텍스트'}</p>
        {kind === 'formula' && segmentId !== undefined ? (
          <img
            className="explain-query-img"
            src={segmentPngUrl(docId, segmentId)}
            alt="선택한 수식"
          />
        ) : (
          <p className="explain-query">{text}</p>
        )}
        {status === 'error' ? (
          <p className="explain-error">{errorMsg}</p>
        ) : (
          <>
            {answer !== '' && <p className="explain-answer">{answer}</p>}
            {status === 'streaming' && <p className="explain-status">설명 생성 중…</p>}
            {status === 'done' && answer === '' && (
              <p className="explain-status">설명이 제공되지 않았습니다.</p>
            )}
          </>
        )}
      </div>
    </aside>
  )
}

export default ExplainPanel
