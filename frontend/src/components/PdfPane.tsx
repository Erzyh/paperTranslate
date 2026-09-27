import { useEffect, useRef, useState } from 'react'
import type { RefObject } from 'react'
import type { RenderTask } from 'pdfjs-dist'
import { TextLayer, loadPdf } from '../pdf'
import type { PDFDocumentProxy } from '../pdf'
import type { AssetMention, SegmentInfo } from '../api'

const MAX_PAGE_WIDTH = 860
const PANE_GUTTER = 48

interface PaneProps {
  url: string
  label: string
  segmentsByPage: Map<number, SegmentInfo[]>
  mentionsByPage: Map<number, AssetMention[]>
  hoverId: string | null
  selectedId: string | null
  onSegHover: (id: string | null) => void
  onSegClick: (id: string) => void
  onMentionHover: (mention: AssetMention, x: number, y: number) => void
  onMentionLeave: () => void
  onMentionClick: (mention: AssetMention, x: number, y: number) => void
  scrollElRef: RefObject<HTMLDivElement | null>
  onScroll: () => void
  onPointerEnter: () => void
  /** Render a transparent pdfjs text layer so the user can drag-select text. */
  textSelectable?: boolean
}

interface PageProps {
  pdf: PDFDocumentProxy
  pageNumber: number // 1-based
  width: number
  segments: SegmentInfo[]
  mentions: AssetMention[]
  hoverId: string | null
  selectedId: string | null
  onSegHover: (id: string | null) => void
  onSegClick: (id: string) => void
  onMentionHover: (mention: AssetMention, x: number, y: number) => void
  onMentionLeave: () => void
  onMentionClick: (mention: AssetMention, x: number, y: number) => void
  textSelectable: boolean
}

function PdfPageView({
  pdf,
  pageNumber,
  width,
  segments,
  mentions,
  hoverId,
  selectedId,
  onSegHover,
  onSegClick,
  onMentionHover,
  onMentionLeave,
  onMentionClick,
  textSelectable,
}: PageProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const textLayerRef = useRef<HTMLDivElement>(null)
  const [layout, setLayout] = useState<{ scale: number; height: number } | null>(null)

  useEffect(() => {
    let cancelled = false
    let renderTask: RenderTask | null = null
    let textLayer: TextLayer | null = null

    const render = async () => {
      const page = await pdf.getPage(pageNumber)
      if (cancelled) return
      const baseViewport = page.getViewport({ scale: 1 })
      const scale = width / baseViewport.width
      const viewport = page.getViewport({ scale })
      const canvas = canvasRef.current
      if (!canvas) return
      const dpr = Math.min(window.devicePixelRatio || 1, 2)
      canvas.width = Math.floor(viewport.width * dpr)
      canvas.height = Math.floor(viewport.height * dpr)
      canvas.style.width = `${Math.floor(viewport.width)}px`
      canvas.style.height = `${Math.floor(viewport.height)}px`
      setLayout({ scale, height: viewport.height })
      renderTask = page.render({
        canvas,
        viewport,
        transform: dpr !== 1 ? [dpr, 0, 0, dpr, 0, 0] : undefined,
      })
      const textLayerDiv = textLayerRef.current
      if (textLayerDiv) {
        // pdfjs positions the spans with CSS relative to --total-scale-factor.
        textLayerDiv.replaceChildren()
        textLayerDiv.style.setProperty('--total-scale-factor', String(scale))
        textLayer = new TextLayer({
          textContentSource: page.streamTextContent(),
          container: textLayerDiv,
          viewport,
        })
        // A text-layer failure must not break the canvas rendering.
        textLayer.render().catch(() => {})
      }
      await renderTask.promise
    }

    render().catch(() => {
      // Rendering was cancelled or the document was destroyed; nothing to do.
    })

    return () => {
      cancelled = true
      renderTask?.cancel()
      textLayer?.cancel()
    }
  }, [pdf, pageNumber, width])

  const scale = layout?.scale ?? width / 612 // fallback: US Letter width in points
  const height = layout?.height ?? width * 1.294

  return (
    <div className="pdf-page" style={{ width, height }}>
      <canvas ref={canvasRef} />
      {textSelectable && <div className="text-layer" ref={textLayerRef} />}
      {layout !== null &&
        segments.map((seg) => {
          const [x0, y0, x1, y1] = seg.bbox
          const active = seg.seg_id === hoverId || seg.seg_id === selectedId
          const formula = seg.kind === 'formula'
          return (
            <div
              key={seg.seg_id}
              className={`seg-overlay${active ? ' active' : ''}${formula ? ' seg-formula' : ''}`}
              style={{
                left: x0 * scale,
                top: y0 * scale,
                width: (x1 - x0) * scale,
                height: (y1 - y0) * scale,
              }}
              onMouseEnter={() => onSegHover(seg.seg_id)}
              onMouseLeave={() => onSegHover(null)}
              onClick={() => onSegClick(seg.seg_id)}
            />
          )
        })}
      {layout !== null &&
        mentions.map((mention, idx) => {
          const [x0, y0, x1, y1] = mention.bbox
          return (
            <div
              key={`${mention.key}-${idx}`}
              className={`mention-overlay${mention.kind === 'cite' ? ' mention-cite' : ''}`}
              style={{
                left: x0 * scale,
                top: y0 * scale,
                width: (x1 - x0) * scale,
                height: (y1 - y0) * scale,
              }}
              onMouseEnter={(e) => onMentionHover(mention, e.clientX, e.clientY)}
              onMouseLeave={onMentionLeave}
              onClick={(e) => onMentionClick(mention, e.clientX, e.clientY)}
            />
          )
        })}
    </div>
  )
}

function PdfPane({
  url,
  label,
  segmentsByPage,
  mentionsByPage,
  hoverId,
  selectedId,
  onSegHover,
  onSegClick,
  onMentionHover,
  onMentionLeave,
  onMentionClick,
  scrollElRef,
  onScroll,
  onPointerEnter,
  textSelectable = false,
}: PaneProps) {
  const [pdf, setPdf] = useState<PDFDocumentProxy | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [pageWidth, setPageWidth] = useState(0)

  useEffect(() => {
    let disposed = false
    const task = loadPdf(url)
    task.promise
      .then((doc) => {
        if (disposed) return
        setPdf(doc)
      })
      .catch(() => {
        if (!disposed) setError('PDF를 불러오지 못했습니다.')
      })
    return () => {
      disposed = true
      setPdf(null)
      task.cancel() // destroys the loading task and any loaded document
    }
  }, [url])

  useEffect(() => {
    const el = scrollElRef.current
    if (!el) return
    const update = () => {
      const width = Math.min(MAX_PAGE_WIDTH, el.clientWidth - PANE_GUTTER)
      setPageWidth(Math.max(0, Math.floor(width)))
    }
    update()
    const observer = new ResizeObserver(update)
    observer.observe(el)
    return () => observer.disconnect()
  }, [scrollElRef])

  return (
    <section className="pane" onMouseEnter={onPointerEnter}>
      <div className="pane-label">{label}</div>
      <div className="pane-scroll" ref={scrollElRef} onScroll={onScroll}>
        {error && <div className="viewer-message">{error}</div>}
        {!error && !pdf && <div className="viewer-message">PDF 불러오는 중…</div>}
        {pdf !== null &&
          pageWidth > 0 &&
          Array.from({ length: pdf.numPages }, (_, i) => (
            <PdfPageView
              key={i + 1}
              pdf={pdf}
              pageNumber={i + 1}
              width={pageWidth}
              segments={segmentsByPage.get(i) ?? []}
              mentions={mentionsByPage.get(i) ?? []}
              hoverId={hoverId}
              selectedId={selectedId}
              onSegHover={onSegHover}
              onSegClick={onSegClick}
              onMentionHover={onMentionHover}
              onMentionLeave={onMentionLeave}
              onMentionClick={onMentionClick}
              textSelectable={textSelectable}
            />
          ))}
      </div>
    </section>
  )
}

export default PdfPane
