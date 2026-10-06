// REST API client for the paperTranslate backend.
// The request/response shapes below mirror the backend routes exactly.

// The built frontend is served by the backend itself (desktop app or
// `uvicorn` on any port), so requests go to the same origin. Only the Vite dev
// server (npm run dev) runs separately and talks to the backend on :8000.
export const API_BASE = import.meta.env.DEV
  ? `http://${window.location.hostname}:8000`
  : window.location.origin

export type DocumentStatus = 'uploaded' | 'translating' | 'done' | 'error' | 'cancelled'

/** Fine-grained stage of a run (batch board); terminal statuses repeat here. */
export type JobPhase =
  | 'queued'
  | 'analyzing'
  | 'glossary'
  | 'translating'
  | 'rendering'
  | 'done'
  | 'error'
  | 'cancelled'
  | 'uploaded'

export type SegmentKind =
  | 'body'
  | 'heading'
  | 'caption'
  | 'header_footer'
  | 'formula'
  | 'author'
  | 'reference'

export type ExplainKind = 'selection' | 'formula'

/** Response of POST /api/documents */
export interface DocumentInfo {
  id: string
  filename: string
  num_pages: number
  status: DocumentStatus
}

/** Response of GET /api/documents/{id} */
export interface DocumentDetail extends DocumentInfo {
  progress: { done: number; total: number; page?: number }
  error: string | null
  phase: JobPhase
  engine: 'ollama' | 'openai' | 'stub' | null
  /** 1-based position while waiting for an engine slot, else null */
  queue_position: number | null
  /** seconds spent in the translating stage (ETA), else null */
  elapsed: number | null
}

/** Response of GET /api/documents?ids=... */
export interface BatchStatus {
  documents: DocumentDetail[]
  missing: string[]
  queue: {
    running: Record<string, number>
    queued: number
    limits: Record<string, number>
  }
}

/** Item of GET /api/documents/{id}/segments */
export interface SegmentInfo {
  seg_id: string
  page: number
  bbox: [number, number, number, number]
  /** Where the translation landed in the output PDF (paragraphs flow within
   *  their column); null for older documents and untranslated segments. */
  translated_bbox?: [number, number, number, number] | null
  kind: SegmentKind
  source: string
  translated: string | null
}

/** Payload of SSE "progress" events (runner progress callback dict) */
export interface TranslateProgress {
  segment_id: string
  done: number
  total: number
  page: number
  num_pages: number
  source_preview: string
  translated_preview: string
}

/** Payload of SSE "error" events */
export interface TranslateError {
  detail: string
}

export type AssetMentionKind = 'figure' | 'table' | 'cite'
export type AssetMentionSide = 'original' | 'translated'

/** Item of GET /api/documents/{id}/assets "figures" */
export interface AssetFigure {
  key: string // "figure-1" | "table-2" ...
  page: number
  bbox: [number, number, number, number]
  caption: string
}

/** Item of GET /api/documents/{id}/assets "mentions" */
export interface AssetMention {
  key: string
  kind: AssetMentionKind
  side: AssetMentionSide
  page: number
  bbox: [number, number, number, number]
  ref: number | null
}

/** Value of GET /api/documents/{id}/assets "references" (keyed by citation number) */
export interface ReferenceEntry {
  entry: string
  title: string | null
}

/** Response of GET /api/documents/{id}/assets */
export interface DocumentAssets {
  figures: AssetFigure[]
  mentions: AssetMention[]
  references: Record<string, ReferenceEntry>
}

/** Response of GET /api/documents/{id}/references/{n}/details */
export interface ReferenceDetails {
  found: boolean
  title: string | null
  authors: string[]
  year: number | null
  abstract: string | null
  url: string | null
}

async function parseJsonOrThrow<T>(res: Response, failMessage: string): Promise<T> {
  if (!res.ok) {
    let detail = ''
    try {
      const body = (await res.json()) as { detail?: string }
      if (body && typeof body.detail === 'string') detail = ` (${body.detail})`
    } catch {
      // ignore non-JSON error bodies
    }
    throw new Error(`${failMessage}${detail}`)
  }
  return (await res.json()) as T
}

/** POST /api/documents — upload + synchronous layout analysis. */
export async function uploadDocument(file: File): Promise<DocumentInfo> {
  const form = new FormData()
  form.append('file', file)
  const res = await fetch(`${API_BASE}/api/documents`, { method: 'POST', body: form })
  return parseJsonOrThrow<DocumentInfo>(res, '업로드에 실패했습니다')
}

/** GET /api/documents/{id} */
export async function getDocument(id: string): Promise<DocumentDetail> {
  const res = await fetch(`${API_BASE}/api/documents/${id}`)
  return parseJsonOrThrow<DocumentDetail>(res, '문서 정보를 가져오지 못했습니다')
}

/**
 * POST /api/documents/{id}/translate — starts the background pipeline.
 * With apiKey the job uses the API model apiModel (OpenAI, Gemini or Claude —
 * the backend picks the provider from the model; backend default when
 * omitted); otherwise the local Ollama translator (ollamaModel, or the
 * configured default). The key is sent in the request body; the backend
 * never persists it (the browser may keep it in localStorage at the user's
 * request, see BatchPage).
 */
export async function startTranslation(
  id: string,
  opts: { apiKey?: string; apiModel?: string; ollamaModel?: string } = {},
): Promise<{ status: string }> {
  const body = opts.apiKey
    ? { api_key: opts.apiKey, api_model: opts.apiModel ?? null }
    : { ollama_model: opts.ollamaModel ?? null }
  const res = await fetch(`${API_BASE}/api/documents/${id}/translate`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  })
  return parseJsonOrThrow<{ status: string }>(res, '번역을 시작하지 못했습니다')
}

/** Response of GET /api/local-models — installed Ollama chat models. */
export interface LocalModels {
  /** false when Ollama is not running (models is then empty) */
  available: boolean
  models: { name: string; size_gb: number }[]
  /** engine default used when a job names no model */
  default: string
}

/** GET /api/local-models */
export async function fetchLocalModels(): Promise<LocalModels> {
  const res = await fetch(`${API_BASE}/api/local-models`)
  return parseJsonOrThrow<LocalModels>(res, '로컬 모델 목록을 가져오지 못했습니다')
}

/** GET /api/documents?ids=a,b — status of many documents in one request. */
export async function getBatchStatus(ids: string[]): Promise<BatchStatus> {
  const res = await fetch(`${API_BASE}/api/documents?ids=${encodeURIComponent(ids.join(','))}`)
  return parseJsonOrThrow<BatchStatus>(res, '상태를 확인하지 못했습니다')
}

/** POST /api/documents/{id}/cancel — queued: dropped now; running: stops at the next segment. */
export async function cancelTranslation(id: string): Promise<{ status: string }> {
  const res = await fetch(`${API_BASE}/api/documents/${id}/cancel`, { method: 'POST' })
  return parseJsonOrThrow<{ status: string }>(res, '취소하지 못했습니다')
}

/** GET /api/documents/{id}/segments?page=n (page omitted -> all pages) */
export async function fetchSegments(id: string, page?: number): Promise<SegmentInfo[]> {
  const query = page === undefined ? '' : `?page=${page}`
  const res = await fetch(`${API_BASE}/api/documents/${id}/segments${query}`)
  return parseJsonOrThrow<SegmentInfo[]>(res, '세그먼트 정보를 가져오지 못했습니다')
}

/** GET /api/documents/{id}/assets — figures/tables, mentions, references. */
export async function fetchAssets(id: string): Promise<DocumentAssets> {
  const res = await fetch(`${API_BASE}/api/documents/${id}/assets`)
  return parseJsonOrThrow<DocumentAssets>(res, '그림/표 정보를 가져오지 못했습니다')
}

/** GET /api/documents/{id}/references/{n}/details — external metadata lookup. */
export async function fetchReferenceDetails(id: string, n: number): Promise<ReferenceDetails> {
  const res = await fetch(`${API_BASE}/api/documents/${id}/references/${n}/details`)
  return parseJsonOrThrow<ReferenceDetails>(res, '참고문헌 정보를 가져오지 못했습니다')
}

/** GET /api/documents/{id}/figures/{key}.png — cropped figure/table render. */
export function figurePngUrl(id: string, key: string): string {
  return `${API_BASE}/api/documents/${id}/figures/${encodeURIComponent(key)}.png`
}

/** GET /api/documents/{id}/segments/{segId}.png — cropped segment render. */
export function segmentPngUrl(id: string, segId: string): string {
  return `${API_BASE}/api/documents/${id}/segments/${encodeURIComponent(segId)}.png`
}

/**
 * POST /api/documents/{id}/explain — streams the answer as text/plain.
 * Returns the raw Response so the caller can read the body incrementally.
 * With segmentId set, the backend re-extracts the model input from the
 * segment's region in the source PDF and ignores text.
 */
export function requestExplain(
  id: string,
  text: string,
  kind: ExplainKind,
  signal?: AbortSignal,
  segmentId?: string,
): Promise<Response> {
  return fetch(`${API_BASE}/api/documents/${id}/explain`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ text, kind, segment_id: segmentId ?? null }),
    signal,
  })
}

/** SSE endpoint URL for EventSource. */
export function eventsUrl(id: string): string {
  return `${API_BASE}/api/documents/${id}/events`
}

export function originalPdfUrl(id: string): string {
  return `${API_BASE}/api/documents/${id}/original.pdf`
}

export function outputPdfUrl(id: string): string {
  return `${API_BASE}/api/documents/${id}/output.pdf`
}

/**
 * POST /api/archives — zip the translated PDFs of finished documents.
 * ``path`` is where each PDF goes inside the zip (keeps folder structure).
 * Returns the absolute download URL of the zip.
 */
export async function createArchive(
  items: { id: string; path?: string }[],
): Promise<{ url: string; count: number }> {
  const res = await fetch(`${API_BASE}/api/archives`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ items }),
  })
  const body = await parseJsonOrThrow<{ url: string; count: number }>(
    res,
    '전체 다운로드 파일을 만들지 못했습니다',
  )
  return { ...body, url: `${API_BASE}${body.url}` }
}

// ---------------------------------------------------------------- app update

/** Response of GET /api/app/update */
export interface UpdateInfo {
  current: string
  /** false when running from source (updates only apply to the packaged app) */
  enabled: boolean
  available: boolean
  latest?: string
  /** the user declined this version — do not offer it again */
  skipped?: boolean
  notes?: string
  page?: string
  size?: number
  error?: string
}

/** Response of GET /api/app/update/status */
export interface UpdateStatus {
  state: 'idle' | 'downloading' | 'ready' | 'error'
  version: string | null
  progress: number
  error: string | null
}

export async function checkUpdate(): Promise<UpdateInfo> {
  const res = await fetch(`${API_BASE}/api/app/update`)
  return parseJsonOrThrow<UpdateInfo>(res, '업데이트를 확인하지 못했습니다')
}

export async function skipUpdate(version: string): Promise<void> {
  const res = await fetch(`${API_BASE}/api/app/update/skip`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ version }),
  })
  await parseJsonOrThrow(res, '건너뛰기를 저장하지 못했습니다')
}

export async function downloadUpdate(): Promise<UpdateStatus> {
  const res = await fetch(`${API_BASE}/api/app/update/download`, { method: 'POST' })
  return parseJsonOrThrow<UpdateStatus>(res, '업데이트를 받지 못했습니다')
}

export async function getUpdateStatus(): Promise<UpdateStatus> {
  const res = await fetch(`${API_BASE}/api/app/update/status`)
  return parseJsonOrThrow<UpdateStatus>(res, '업데이트 상태를 확인하지 못했습니다')
}

/** Close and reopen the app so a downloaded update installs now. */
export async function restartApp(): Promise<void> {
  const res = await fetch(`${API_BASE}/api/app/restart`, { method: 'POST' })
  await parseJsonOrThrow(res, '재시작하지 못했습니다')
}
