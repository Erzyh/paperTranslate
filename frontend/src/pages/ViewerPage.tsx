import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { fetchAssets, fetchSegments, originalPdfUrl, outputPdfUrl } from '../api'
import type {
  AssetMention,
  DocumentAssets,
  DocumentInfo,
  ExplainKind,
  ReferenceDetails,
  SegmentInfo,
} from '../api'
import PdfPane from '../components/PdfPane'
import FigureTooltip from '../components/FigureTooltip'
import CitePopup from '../components/CitePopup'
import ExplainPanel from '../components/ExplainPanel'

// Minimum selected characters before the "AI 설명" mini button appears.
const MIN_SELECTION_LENGTH = 8
// Gap between the selection rectangle and the mini button.
const SELECTION_BUTTON_GAP = 8
// Horizontal clearance that keeps the mini button inside the viewport.
const SELECTION_BUTTON_MARGIN = 60

interface Props {
  doc: DocumentInfo
  onReset: () => void
}

type PaneSide = 'left' | 'right'

interface FigTooltipState {
  key: string
  x: number
  y: number
}

interface CitePopupState {
  refNum: number
  x: number
  y: number
}

interface ExplainRequest {
  /** Monotonic counter so every request remounts the panel via a React key. */
  seq: number
  text: string
  kind: ExplainKind
  /** Set for formula segments: the backend extracts the text server-side. */
  segmentId?: string
}

interface SelectionButtonState {
  text: string
  /** Viewport coordinates of the selection edge the button attaches to. */
  x: number
  y: number
}

function ViewerPage({ doc, onReset }: Props) {
  const [segments, setSegments] = useState<SegmentInfo[]>([])
  const [assets, setAssets] = useState<DocumentAssets | null>(null)
  const [hoverId, setHoverId] = useState<string | null>(null)
  const [selectedId, setSelectedId] = useState<string | null>(null)
  const [figTooltip, setFigTooltip] = useState<FigTooltipState | null>(null)
  const [citePopup, setCitePopup] = useState<CitePopupState | null>(null)
  const [splitView, setSplitView] = useState(true)
  const [explainReq, setExplainReq] = useState<ExplainRequest | null>(null)
  const [explainBusy, setExplainBusy] = useState(false)
  const [selButton, setSelButton] = useState<SelectionButtonState | null>(null)

  const leftRef = useRef<HTMLDivElement | null>(null)
  const rightRef = useRef<HTMLDivElement | null>(null)
  // The pane the pointer is over drives scrolling; the other pane follows.
  const driverRef = useRef<PaneSide | null>(null)
  // Reference details cache ("docId:refNum" -> details) so re-clicking the
  // same citation shows the result instantly without another request.
  const detailsCacheRef = useRef(new Map<string, ReferenceDetails>())

  useEffect(() => {
    let disposed = false
    fetchSegments(doc.id)
      .then((items) => {
        if (!disposed) setSegments(items)
      })
      .catch(() => {
        // Overlays are optional; the PDFs still render without them.
      })
    return () => {
      disposed = true
    }
  }, [doc.id])

  useEffect(() => {
    let disposed = false
    // The viewer only mounts once the document is done, so one fetch suffices.
    fetchAssets(doc.id)
      .then((data) => {
        if (!disposed) setAssets(data)
      })
      .catch(() => {
        // Mention overlays are optional; the PDFs still render without them.
      })
    return () => {
      disposed = true
    }
  }, [doc.id])

  const segmentById = useMemo(() => {
    const byId = new Map<string, SegmentInfo>()
    for (const seg of segments) byId.set(seg.seg_id, seg)
    return byId
  }, [segments])

  const segmentsByPage = useMemo(() => {
    const byPage = new Map<number, SegmentInfo[]>()
    for (const seg of segments) {
      const list = byPage.get(seg.page)
      if (list) list.push(seg)
      else byPage.set(seg.page, [seg])
    }
    return byPage
  }, [segments])

  // Mentions split per pane: the original pane shows side=original boxes and
  // the translated pane shows side=translated boxes.
  const mentionsBySide = useMemo(() => {
    const bySide = {
      original: new Map<number, AssetMention[]>(),
      translated: new Map<number, AssetMention[]>(),
    }
    for (const mention of assets?.mentions ?? []) {
      const byPage = bySide[mention.side]
      if (!byPage) continue
      const list = byPage.get(mention.page)
      if (list) list.push(mention)
      else byPage.set(mention.page, [mention])
    }
    return bySide
  }, [assets])

  const syncScroll = useCallback((side: PaneSide) => {
    if (driverRef.current !== null && driverRef.current !== side) return
    const src = side === 'left' ? leftRef.current : rightRef.current
    const dst = side === 'left' ? rightRef.current : leftRef.current
    if (!src || !dst) return
    const srcMax = src.scrollHeight - src.clientHeight
    const dstMax = dst.scrollHeight - dst.clientHeight
    if (srcMax <= 0 || dstMax <= 0) return
    // Proportional mapping between the two documents' scroll ranges.
    dst.scrollTop = (src.scrollTop / srcMax) * dstMax
  }, [])

  const handleLeftScroll = useCallback(() => syncScroll('left'), [syncScroll])
  const handleRightScroll = useCallback(() => syncScroll('right'), [syncScroll])
  const claimLeft = useCallback(() => {
    driverRef.current = 'left'
  }, [])
  const claimRight = useCallback(() => {
    driverRef.current = 'right'
  }, [])

  // Duplicate-request guard: ignore new explain triggers while one streams.
  const openExplain = useCallback(
    (text: string, kind: ExplainKind, segmentId?: string) => {
      if (explainBusy) return
      setExplainReq((prev) => ({ seq: (prev?.seq ?? 0) + 1, text, kind, segmentId }))
    },
    [explainBusy],
  )

  const closeExplain = useCallback(() => {
    setExplainReq(null)
  }, [])

  const handleSegClick = useCallback(
    (id: string) => {
      const seg = segmentById.get(id)
      if (seg?.kind === 'formula') {
        // The server re-extracts the formula text from the PDF region, so no
        // client-side text is needed (seg.source may hold ⟦EQn⟧ tokens).
        openExplain('', 'formula', seg.seg_id)
        return
      }
      setSelectedId((prev) => (prev === id ? null : id))
    },
    [segmentById, openExplain],
  )

  // Watch text selections inside the original pane and float the mini button
  // next to the selection once it is long enough.
  useEffect(() => {
    const update = () => {
      const container = leftRef.current
      if (!container) {
        setSelButton(null)
        return
      }
      const sel = document.getSelection()
      if (!sel || sel.rangeCount === 0 || sel.isCollapsed) {
        setSelButton(null)
        return
      }
      const range = sel.getRangeAt(0)
      if (!container.contains(range.commonAncestorContainer)) {
        setSelButton(null)
        return
      }
      const text = sel.toString().replace(/\s+/g, ' ').trim()
      if (text.length < MIN_SELECTION_LENGTH) {
        setSelButton(null)
        return
      }
      const rect = range.getBoundingClientRect()
      if (rect.width === 0 && rect.height === 0) {
        setSelButton(null)
        return
      }
      setSelButton({ text, x: rect.left + rect.width / 2, y: rect.bottom })
    }
    document.addEventListener('selectionchange', update)
    // Reposition (or hide) the fixed button when the pane scrolls under it.
    const scrollEl = leftRef.current
    scrollEl?.addEventListener('scroll', update)
    return () => {
      document.removeEventListener('selectionchange', update)
      scrollEl?.removeEventListener('scroll', update)
    }
  }, [splitView])

  const handleSelectionExplain = useCallback(() => {
    if (selButton === null) return
    openExplain(selButton.text, 'selection')
    setSelButton(null)
  }, [selButton, openExplain])

  const handleMentionHover = useCallback((mention: AssetMention, x: number, y: number) => {
    // Only figure/table mentions get an image preview tooltip.
    if (mention.kind === 'cite') return
    setFigTooltip({ key: mention.key, x, y })
  }, [])

  const handleMentionLeave = useCallback(() => {
    setFigTooltip(null)
  }, [])

  const handleMentionClick = useCallback((mention: AssetMention, x: number, y: number) => {
    if (mention.kind !== 'cite' || mention.ref === null) return
    setCitePopup({ refNum: mention.ref, x, y })
  }, [])

  const closeCitePopup = useCallback(() => {
    setCitePopup(null)
  }, [])

  return (
    <div className="page">
      <header className="topbar">
        <span className="filename">{doc.filename}</span>
        <div className="topbar-actions">
          <button type="button" className="btn" onClick={() => setSplitView((v) => !v)}>
            {splitView ? '단일 보기' : '분할 보기'}
          </button>
          <button type="button" className="btn" onClick={onReset}>
            목록으로
          </button>
          <a className="btn btn-accent" href={outputPdfUrl(doc.id)} download>
            PDF 다운로드
          </a>
        </div>
      </header>
      <div className="viewer-body">
        {splitView && (
          <PdfPane
            url={originalPdfUrl(doc.id)}
            label="원문"
            segmentsByPage={segmentsByPage}
            mentionsByPage={mentionsBySide.original}
            hoverId={hoverId}
            selectedId={selectedId}
            onSegHover={setHoverId}
            onSegClick={handleSegClick}
            onMentionHover={handleMentionHover}
            onMentionLeave={handleMentionLeave}
            onMentionClick={handleMentionClick}
            scrollElRef={leftRef}
            onScroll={handleLeftScroll}
            onPointerEnter={claimLeft}
            textSelectable
          />
        )}
        <PdfPane
          url={outputPdfUrl(doc.id)}
          label="번역"
          segmentsByPage={segmentsByPage}
          mentionsByPage={mentionsBySide.translated}
          hoverId={hoverId}
          selectedId={selectedId}
          onSegHover={setHoverId}
          onSegClick={handleSegClick}
          onMentionHover={handleMentionHover}
          onMentionLeave={handleMentionLeave}
          onMentionClick={handleMentionClick}
          scrollElRef={rightRef}
          onScroll={handleRightScroll}
          onPointerEnter={claimRight}
        />
        {explainReq !== null && (
          <ExplainPanel
            key={explainReq.seq}
            docId={doc.id}
            text={explainReq.text}
            kind={explainReq.kind}
            segmentId={explainReq.segmentId}
            onBusyChange={setExplainBusy}
            onClose={closeExplain}
          />
        )}
      </div>
      {selButton !== null && (
        <button
          type="button"
          className="btn btn-outline-accent explain-mini-btn"
          style={{
            left: Math.min(
              Math.max(selButton.x, SELECTION_BUTTON_MARGIN),
              window.innerWidth - SELECTION_BUTTON_MARGIN,
            ),
            top: selButton.y + SELECTION_BUTTON_GAP,
          }}
          disabled={explainBusy}
          // Keep the text selection alive when the button is pressed.
          onMouseDown={(e) => e.preventDefault()}
          onClick={handleSelectionExplain}
        >
          AI 설명
        </button>
      )}
      {figTooltip !== null && (
        <FigureTooltip
          key={figTooltip.key}
          docId={doc.id}
          figKey={figTooltip.key}
          x={figTooltip.x}
          y={figTooltip.y}
        />
      )}
      {citePopup !== null && (
        <CitePopup
          key={citePopup.refNum}
          docId={doc.id}
          refNum={citePopup.refNum}
          entry={assets?.references[String(citePopup.refNum)]?.entry ?? null}
          x={citePopup.x}
          y={citePopup.y}
          cache={detailsCacheRef.current}
          onClose={closeCitePopup}
        />
      )}
    </div>
  )
}

export default ViewerPage
