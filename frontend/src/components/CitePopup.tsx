import { useEffect, useLayoutEffect, useRef, useState } from 'react'
import { fetchReferenceDetails } from '../api'
import type { ReferenceDetails } from '../api'

// Popup card for a citation mention. The parent remounts this component (via a
// React key) whenever the citation number changes, so all state below always
// belongs to the current reference.

const CLICK_OFFSET = 10
const VIEWPORT_MARGIN = 8
const MAX_AUTHORS = 3

interface Props {
  docId: string
  refNum: number
  /** Local bibliography entry text (always shown when available). */
  entry: string | null
  /** Click position in viewport coordinates. */
  x: number
  y: number
  /** Shared details cache, keyed by "docId:refNum". Owned by the parent. */
  cache: Map<string, ReferenceDetails>
  onClose: () => void
}

function formatAuthors(authors: string[]): string | null {
  if (authors.length === 0) return null
  const shown = authors.slice(0, MAX_AUTHORS).join(', ')
  const rest = authors.length - MAX_AUTHORS
  return rest > 0 ? `${shown} 외 ${rest}명` : shown
}

function CitePopup({ docId, refNum, entry, x, y, cache, onClose }: Props) {
  const boxRef = useRef<HTMLDivElement>(null)
  const abstractRef = useRef<HTMLParagraphElement>(null)
  const [details, setDetails] = useState<ReferenceDetails | null>(null)
  const [failed, setFailed] = useState(false)
  const [abstractOpen, setAbstractOpen] = useState(false)
  const [abstractClamped, setAbstractClamped] = useState(false)
  const [pos, setPos] = useState<{ left: number; top: number } | null>(null)

  // Fetch details once per reference; re-clicks hit the parent-owned cache.
  useEffect(() => {
    const cacheKey = `${docId}:${refNum}`
    const cached = cache.get(cacheKey)
    if (cached) {
      setDetails(cached)
      return
    }
    let disposed = false
    fetchReferenceDetails(docId, refNum)
      .then((data) => {
        cache.set(cacheKey, data)
        if (!disposed) setDetails(data)
      })
      .catch(() => {
        // Network failure is shown the same way as found=false.
        if (!disposed) setFailed(true)
      })
    return () => {
      disposed = true
    }
  }, [docId, refNum, cache])

  // Close on outside click or ESC. The opening click's mousedown happened
  // before this effect ran, so it never closes the popup it just opened.
  useEffect(() => {
    const onDocMouseDown = (event: MouseEvent) => {
      const el = boxRef.current
      if (el && event.target instanceof Node && el.contains(event.target)) return
      onClose()
    }
    const onKeyDown = (event: KeyboardEvent) => {
      if (event.key === 'Escape') onClose()
    }
    document.addEventListener('mousedown', onDocMouseDown)
    document.addEventListener('keydown', onKeyDown)
    return () => {
      document.removeEventListener('mousedown', onDocMouseDown)
      document.removeEventListener('keydown', onKeyDown)
    }
  }, [onClose])

  // Keep the card inside the viewport. Re-runs whenever the content (and thus
  // the card size) changes: loading -> details, abstract toggling, etc.
  useLayoutEffect(() => {
    const el = boxRef.current
    if (!el) return
    const rect = el.getBoundingClientRect()
    const left = Math.max(
      VIEWPORT_MARGIN,
      Math.min(x + CLICK_OFFSET, window.innerWidth - rect.width - VIEWPORT_MARGIN),
    )
    const top = Math.max(
      VIEWPORT_MARGIN,
      Math.min(y + CLICK_OFFSET, window.innerHeight - rect.height - VIEWPORT_MARGIN),
    )
    setPos({ left, top })
  }, [x, y, details, failed, abstractOpen])

  // Only offer the "more" toggle when the clamped abstract actually overflows.
  useLayoutEffect(() => {
    if (abstractOpen) return
    const el = abstractRef.current
    if (!el) return
    setAbstractClamped(el.scrollHeight > el.clientHeight + 1)
  }, [details, abstractOpen])

  const authors = details?.found ? formatAuthors(details.authors) : null
  const meta = details?.found
    ? [authors, details.year !== null ? String(details.year) : null].filter(Boolean).join(' · ')
    : ''

  return (
    <div
      ref={boxRef}
      className="cite-popup"
      style={{
        left: pos ? pos.left : x + CLICK_OFFSET,
        top: pos ? pos.top : y + CLICK_OFFSET,
        visibility: pos !== null ? 'visible' : 'hidden',
      }}
    >
      <div className="cite-popup-head">
        <span className="cite-popup-num">[{refNum}]</span>
        {entry !== null && <p className="cite-popup-entry">{entry}</p>}
      </div>
      <div className="cite-popup-details">
        {details === null && !failed && (
          <p className="cite-popup-status">논문 정보 조회 중…</p>
        )}
        {(failed || (details !== null && !details.found)) && (
          <p className="cite-popup-status">외부 정보를 찾지 못했습니다</p>
        )}
        {details !== null && details.found && (
          <>
            {details.title !== null && <p className="cite-popup-title">{details.title}</p>}
            {meta !== '' && <p className="cite-popup-meta">{meta}</p>}
            {details.abstract !== null && (
              <>
                <p
                  ref={abstractRef}
                  className={`cite-popup-abstract${abstractOpen ? '' : ' clamped'}`}
                >
                  {details.abstract}
                </p>
                {(abstractOpen || abstractClamped) && (
                  <button
                    type="button"
                    className="cite-popup-toggle"
                    onClick={() => setAbstractOpen((v) => !v)}
                  >
                    {abstractOpen ? '접기' : '더보기'}
                  </button>
                )}
              </>
            )}
            {details.url !== null && (
              <a
                className="cite-popup-link"
                href={details.url}
                target="_blank"
                rel="noreferrer"
              >
                원문 보기
              </a>
            )}
          </>
        )}
      </div>
    </div>
  )
}

export default CitePopup
