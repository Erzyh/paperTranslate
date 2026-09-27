// pdfjs-dist setup: register the worker bundle through Vite's ?url import
// so it is served/bundled as a static asset.
import { GlobalWorkerOptions, getDocument } from 'pdfjs-dist'
import type { PDFDocumentProxy } from 'pdfjs-dist'
import pdfWorkerUrl from 'pdfjs-dist/build/pdf.worker.min.mjs?url'

GlobalWorkerOptions.workerSrc = pdfWorkerUrl

/** Load a PDF document. Caller must call destroy() on the returned proxy. */
export function loadPdf(url: string): { promise: Promise<PDFDocumentProxy>; cancel: () => void } {
  const task = getDocument({ url })
  return {
    promise: task.promise,
    cancel: () => {
      void task.destroy()
    },
  }
}

export { TextLayer } from 'pdfjs-dist'
export type { PDFDocumentProxy, PDFPageProxy } from 'pdfjs-dist'
