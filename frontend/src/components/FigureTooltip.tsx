import { useLayoutEffect, useRef, useState } from 'react'
import { figurePngUrl } from '../api'

// Hover preview for a figure/table mention. The parent remounts this component
// (via a React key) whenever the target figure key changes, so load state below
// always belongs to the current image.

const CURSOR_OFFSET = 14
const VIEWPORT_MARGIN = 8

interface Props {
  docId: string
  figKey: string
  /** Cursor position in viewport coordinates at hover time. */
  x: number
  y: number
}

function FigureTooltip({ docId, figKey, x, y }: Props) {
  const boxRef = useRef<HTMLDivElement>(null)
  const [loaded, setLoaded] = useState(false)
  const [failed, setFailed] = useState(false)
  const [pos, setPos] = useState<{ left: number; top: number } | null>(null)

  // Once the image has loaded (its size is known), clamp the tooltip so it
  // never overflows the viewport: flip to the other side of the cursor first,
  // then pin to the margin as a last resort.
  useLayoutEffect(() => {
    if (!loaded) return
    const el = boxRef.current
    if (!el) return
    const rect = el.getBoundingClientRect()
    let left = x + CURSOR_OFFSET
    let top = y + CURSOR_OFFSET
    if (left + rect.width + VIEWPORT_MARGIN > window.innerWidth) {
      left = Math.max(VIEWPORT_MARGIN, x - CURSOR_OFFSET - rect.width)
    }
    if (top + rect.height + VIEWPORT_MARGIN > window.innerHeight) {
      top = Math.max(VIEWPORT_MARGIN, y - CURSOR_OFFSET - rect.height)
    }
    setPos({ left, top })
  }, [loaded, x, y])

  // No tooltip at all when the crop endpoint fails.
  if (failed) return null

  return (
    <div
      ref={boxRef}
      className="fig-tooltip"
      style={{
        left: pos ? pos.left : x + CURSOR_OFFSET,
        top: pos ? pos.top : y + CURSOR_OFFSET,
        visibility: loaded && pos !== null ? 'visible' : 'hidden',
      }}
    >
      <img
        src={figurePngUrl(docId, figKey)}
        alt=""
        onLoad={() => setLoaded(true)}
        onError={() => setFailed(true)}
      />
    </div>
  )
}

export default FigureTooltip
