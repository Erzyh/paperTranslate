import { useCallback, useEffect, useRef, useState } from 'react'
import type { DragEvent } from 'react'
import {
  cancelTranslation,
  fetchLocalModels,
  getBatchStatus,
  outputPdfUrl,
  startTranslation,
  uploadDocument,
} from '../api'
import type { DocumentInfo, LocalModels } from '../api'

// Batch board: many PDFs at once, one card per paper.
// Uploads run UPLOAD_CONCURRENCY at a time; translation order and parallelism
// are decided by the backend queue (per-engine limits). All cards refresh
// from a single GET /api/documents?ids=... per POLL_MS.

const JOBS_KEY = 'papertranslate-jobs'
const MODEL_STORE = 'papertranslate-model'

// API 제공사별 설정. 키는 제공사마다 따로, 사용자가 요청해 이 브라우저에만
// 저장한다 (백엔드는 키를 저장하지 않음). OpenAI 키 저장 이름은 예전 그대로.
type Provider = 'openai' | 'gemini' | 'anthropic'
const PROVIDERS: Record<Provider, { name: string; keyStore: string; keyHint: string }> = {
  openai: { name: 'OpenAI', keyStore: 'papertranslate-openai-key', keyHint: 'sk-...' },
  gemini: { name: 'Gemini', keyStore: 'papertranslate-gemini-key', keyHint: 'AIza...' },
  anthropic: { name: 'Anthropic', keyStore: 'papertranslate-anthropic-key', keyHint: 'sk-ant-...' },
}
const PROVIDER_ORDER: Provider[] = ['openai', 'gemini', 'anthropic']
const PROVIDER_GROUP: Record<Provider, string> = {
  openai: 'OpenAI (API 키 필요)',
  gemini: 'Google Gemini (API 키 필요)',
  anthropic: 'Anthropic Claude (API 키 필요)',
}

// API 번역 모델 (backend/app/config.py 의 API_MODELS 와 같은 목록). 로컬 모델은
// 이 PC의 Ollama에 설치된 것을 불러와 "ollama:<이름>" 값으로 쓴다 (키 불필요).
const API_MODELS: { id: string; label: string; provider: Provider }[] = [
  { id: 'gpt-6-luna', label: 'GPT-6 LUNA', provider: 'openai' },
  { id: 'gpt-5.6-luna', label: 'GPT-5.6 LUNA', provider: 'openai' },
  { id: 'gpt-6-sol', label: 'GPT-6 SOL', provider: 'openai' },
  { id: 'gpt-5.6-terra', label: 'GPT-5.6 TERRA', provider: 'openai' },
  { id: 'gpt-6-astra', label: 'GPT-6 ASTRA', provider: 'openai' },
  { id: 'gemini-3.8-flash', label: 'Gemini 3.8 Flash', provider: 'gemini' },
  { id: 'claude-opus-5-5', label: 'Claude Opus 5.5', provider: 'anthropic' },
]
const LOCAL_PREFIX = 'ollama:'
const DEFAULT_MODEL = 'gpt-5.6-luna'

const isLocalModel = (id: string) => id.startsWith(LOCAL_PREFIX)

function modelLabel(id?: string): string | undefined {
  if (!id) return undefined
  if (isLocalModel(id)) return id.slice(LOCAL_PREFIX.length) || '로컬 기본 모델'
  return API_MODELS.find((m) => m.id === id)?.label
}

// Stored choice → current value format ('qwen' was the old local entry).
function initialModel(saved: string): string {
  if (saved === 'qwen') return LOCAL_PREFIX
  if (isLocalModel(saved) || API_MODELS.some((m) => m.id === saved)) return saved
  return DEFAULT_MODEL
}

function readStored(key: string): string {
  try {
    return localStorage.getItem(key) ?? ''
  } catch {
    return ''
  }
}

function writeStored(key: string, value: string) {
  try {
    if (value) localStorage.setItem(key, value)
    else localStorage.removeItem(key)
  } catch {
    // storage unavailable: the value just isn't remembered
  }
}
const UPLOAD_CONCURRENCY = 3
const POLL_MS = 1500

type JobStatus = 'draft' | 'uploading' | 'translating' | 'done' | 'error' | 'cancelled'

interface Job {
  key: string
  filename: string
  // Folder the file came from (relative, e.g. "papers/2025"), when a folder was added.
  folder?: string
  size?: number
  docId?: string
  numPages?: number
  status: JobStatus
  // Backend phase, plus client-only 'wait-upload' | 'uploading' | 'restoring'.
  phase: string | null
  // Stage the run was in when it stopped (error/cancelled cards).
  lastPhase?: string | null
  engine?: 'local' | 'openai'
  model?: string
  done: number
  total: number
  page?: number
  queuePos?: number | null
  elapsed?: number | null
  error?: string | null
  cancelling?: boolean
}

interface Props {
  onOpen: (doc: DocumentInfo) => void
}

const isActive = (j: Job) => j.status === 'uploading' || j.status === 'translating'
const isFailed = (j: Job) => j.status === 'error' || j.status === 'cancelled'

// A picked file plus its path relative to what the user chose
// ("papers/2025/a.pdf" for a folder, just "a.pdf" for loose files).
interface Picked {
  file: File
  path: string
}

/**
 * Files from a drop, walking into dropped folders (and their subfolders).
 * The entries are read synchronously first: the browser empties the
 * DataTransfer as soon as the drop handler yields.
 */
async function readDropped(dt: DataTransfer): Promise<Picked[]> {
  const entries = Array.from(dt.items ?? [])
    .map((item) => (item.kind === 'file' ? item.webkitGetAsEntry() : null))
    .filter((entry): entry is FileSystemEntry => entry !== null)
  if (!entries.length) return Array.from(dt.files).map((file) => ({ file, path: file.name }))

  const out: Picked[] = []
  const walk = async (entry: FileSystemEntry): Promise<void> => {
    if (entry.isFile) {
      const file = await new Promise<File>((resolve, reject) =>
        (entry as FileSystemFileEntry).file(resolve, reject),
      )
      out.push({ file, path: entry.fullPath.replace(/^\//, '') })
    } else if (entry.isDirectory) {
      const reader = (entry as FileSystemDirectoryEntry).createReader()
      // readEntries returns at most ~100 entries per call; read until empty.
      for (;;) {
        const batch = await new Promise<FileSystemEntry[]>((resolve, reject) =>
          reader.readEntries(resolve, reject),
        )
        if (!batch.length) break
        for (const child of batch) await walk(child)
      }
    }
  }
  for (const entry of entries) await walk(entry)
  return out
}

function folderOf(path: string): string | undefined {
  const cut = path.lastIndexOf('/')
  return cut > 0 ? path.slice(0, cut) : undefined
}

function isPdf(file: File): boolean {
  return file.type === 'application/pdf' || file.name.toLowerCase().endsWith('.pdf')
}

function newKey(): string {
  return crypto.randomUUID ? crypto.randomUUID() : `${Date.now()}-${Math.random()}`
}

// Only uploaded jobs are remembered (ids + names), so cards survive a reload.
function loadJobs(): Job[] {
  try {
    const saved = JSON.parse(localStorage.getItem(JOBS_KEY) || '[]') as Job[]
    return saved
      .filter((j) => j.docId)
      .map((j) => ({ ...j, status: 'translating', phase: 'restoring', done: 0, total: 0 }))
  } catch {
    return []
  }
}

function saveJobs(jobs: Job[]) {
  try {
    const keep = jobs
      .filter((j) => j.docId)
      .map(({ key, filename, folder, docId, engine, model }) => ({
        key,
        filename,
        folder,
        docId,
        engine,
        model,
      }))
    localStorage.setItem(JOBS_KEY, JSON.stringify(keep))
  } catch {
    // storage unavailable: the board just won't survive a reload
  }
}

/** Card percentage: the translating stage dominates, the others get a sliver. */
function jobPercent(j: Job): number {
  const ratio = j.total > 0 ? j.done / j.total : 0
  switch (j.phase) {
    case 'done':
      return 100
    case 'rendering':
      return 96
    case 'translating':
      return 8 + 88 * ratio
    case 'error':
    case 'cancelled':
      return j.total > 0 ? 8 + 88 * ratio : 0
    case 'glossary':
      return 5
    case 'analyzing':
      return 2
    default:
      return 0
  }
}

// A card is just the file name, one progress bar, and the actions for a
// finished paper. Failed papers get a retry button; the reason shows on hover.
function JobCard({
  job,
  onOpen,
  onRetry,
}: {
  job: Job
  onOpen: () => void
  onRetry: () => void
}) {
  const failed = isFailed(job)
  const tone = job.status === 'done' ? 'ok' : failed ? job.status : 'run'
  const path = job.folder ? `${job.folder}/${job.filename}` : job.filename
  return (
    <div className={`job-card job-${job.status}`} title={failed && job.error ? job.error : path}>
      <span className="job-name">{job.filename}</span>
      <div className="job-track">
        <div className={`job-fill job-fill-${tone}`} style={{ width: `${jobPercent(job)}%` }} />
      </div>
      {job.status === 'done' && (
        <div className="job-actions">
          <button type="button" className="btn btn-sm btn-accent" onClick={onOpen}>
            비교 보기
          </button>
          <a className="btn btn-sm" href={outputPdfUrl(job.docId!)} download>
            PDF
          </a>
        </div>
      )}
      {failed && job.docId && (
        <div className="job-actions">
          <button type="button" className="btn btn-sm" onClick={onRetry}>
            다시 번역
          </button>
        </div>
      )}
    </div>
  )
}

function BatchPage({ onOpen }: Props) {
  const inputRef = useRef<HTMLInputElement>(null)
  const [jobs, setJobs] = useState<Job[]>(loadJobs)
  const [dragOver, setDragOver] = useState(false)
  const [notice, setNotice] = useState<string | null>(null)
  // 키·모델은 이 브라우저의 localStorage에 기억한다 (사용자 요청).
  const [apiKeys, setApiKeys] = useState<Record<Provider, string>>(() => ({
    openai: readStored(PROVIDERS.openai.keyStore),
    gemini: readStored(PROVIDERS.gemini.keyStore),
    anthropic: readStored(PROVIDERS.anthropic.keyStore),
  }))
  const [model, setModel] = useState(() => initialModel(readStored(MODEL_STORE)))
  const [showKey, setShowKey] = useState(false)
  const [localModels, setLocalModels] = useState<LocalModels | null>(null)
  const isLocal = isLocalModel(model)
  // Provider of the chosen API model (null for local models).
  const provider: Provider | null = isLocal
    ? null
    : (API_MODELS.find((m) => m.id === model)?.provider ?? 'openai')
  const apiKey = provider ? apiKeys[provider] : ''
  const setApiKey = (value: string) => {
    if (provider) setApiKeys((cur) => ({ ...cur, [provider]: value }))
  }

  // Installed Ollama models for the dropdown. A bare "ollama:" (old 'qwen'
  // choice) resolves to the engine's default model once the list is known.
  useEffect(() => {
    let alive = true
    fetchLocalModels()
      .then((res) => {
        if (!alive) return
        setLocalModels(res)
        setModel((cur) =>
          cur === LOCAL_PREFIX
            ? LOCAL_PREFIX + (res.models.some((m) => m.name === res.default)
              ? res.default
              : (res.models[0]?.name ?? res.default))
            : cur,
        )
      })
      .catch(() => {
        if (alive) setLocalModels({ available: false, models: [], default: '' })
      })
    return () => {
      alive = false
    }
  }, [])

  useEffect(() => {
    for (const p of PROVIDER_ORDER) writeStored(PROVIDERS[p].keyStore, apiKeys[p].trim())
  }, [apiKeys])
  useEffect(() => writeStored(MODEL_STORE, model), [model])
  const jobsRef = useRef(jobs)
  jobsRef.current = jobs
  const filesRef = useRef(new Map<string, File>())
  const folderInputRef = useRef<HTMLInputElement>(null)

  // Folder picker: React has no prop for the non-standard webkitdirectory.
  useEffect(() => {
    folderInputRef.current?.setAttribute('webkitdirectory', '')
  }, [])

  useEffect(() => saveJobs(jobs), [jobs])

  const patchJob = useCallback((key: string, patch: Partial<Job>) => {
    setJobs((list) => list.map((j) => (j.key === key ? { ...j, ...patch } : j)))
  }, [])

  const addFiles = useCallback((picked: Picked[], fromFolder = false) => {
    if (!picked.length) return
    const pdfs = picked
      .filter((p) => isPdf(p.file))
      .sort((a, b) => a.path.localeCompare(b.path, undefined, { numeric: true }))
    const sig = (path: string, size?: number) => `${path}:${size}`
    const seen = new Set(
      jobsRef.current
        .filter((j) => j.status === 'draft')
        .map((j) => sig(j.folder ? `${j.folder}/${j.filename}` : j.filename, j.size)),
    )
    const added: Job[] = []
    for (const { file, path } of pdfs) {
      if (seen.has(sig(path, file.size))) continue // the same file added twice
      seen.add(sig(path, file.size))
      const key = newKey()
      filesRef.current.set(key, file)
      added.push({
        key,
        filename: file.name,
        folder: folderOf(path),
        size: file.size,
        status: 'draft',
        phase: null,
        done: 0,
        total: 0,
      })
    }
    if (added.length) setJobs((prev) => [...added, ...prev])

    const skipped = picked.length - pdfs.length
    const dupes = pdfs.length - added.length
    const notes: string[] = []
    if (fromFolder) {
      notes.push(pdfs.length ? `폴더에서 PDF ${pdfs.length}편을 찾았습니다.` : '선택한 폴더에 PDF가 없습니다.')
      if (skipped) notes.push(`PDF가 아닌 파일 ${skipped}개는 건너뛰었습니다.`)
    } else if (skipped) {
      notes.push(`PDF가 아닌 파일 ${skipped}개는 제외했습니다.`)
    }
    if (dupes) notes.push(`이미 목록에 있는 ${dupes}편은 빼고 넣었습니다.`)
    setNotice(notes.length ? notes.join(' ') : null)
  }, [])

  // One request refreshes every live card.
  useEffect(() => {
    let stop = false
    const tick = async () => {
      const live = jobsRef.current.filter((j) => j.docId && j.status === 'translating')
      if (!live.length) return
      try {
        const res = await getBatchStatus(live.map((j) => j.docId!))
        if (stop) return
        const byId = new Map(res.documents.map((d) => [d.id, d]))
        const missing = new Set(res.missing)
        setJobs((list) =>
          list
            .filter((j) => !(j.docId && missing.has(j.docId)))
            .map((j) => {
              const d = j.docId ? byId.get(j.docId) : undefined
              if (!d || j.status !== 'translating') return j
              const status: JobStatus = d.status === 'uploaded' ? 'error' : d.status
              const stopped = d.phase === 'error' || d.phase === 'cancelled'
              return {
                ...j,
                status,
                phase: d.phase,
                lastPhase: stopped ? (j.lastPhase ?? j.phase) : d.phase,
                done: d.progress.done,
                total: d.progress.total,
                page: d.progress.page,
                numPages: d.num_pages,
                queuePos: d.queue_position,
                elapsed: d.elapsed,
                error: d.status === 'uploaded' ? '번역이 시작되지 않았습니다.' : d.error,
                cancelling: status === 'translating' ? j.cancelling : false,
              }
            }),
        )
      } catch {
        // transient failure: retry on the next tick
      }
    }
    void tick()
    const timer = window.setInterval(() => void tick(), POLL_MS)
    return () => {
      stop = true
      window.clearInterval(timer)
    }
  }, [])

  const run = async (keys: string[]) => {
    const key = isLocal ? undefined : apiKey.trim()
    const apiModel = isLocal ? undefined : model
    const ollamaModel = isLocal ? model.slice(LOCAL_PREFIX.length) || undefined : undefined
    const engine: Job['engine'] = isLocal ? 'local' : 'openai'
    if (provider && !key) {
      setNotice(`맨 위에 ${PROVIDERS[provider].name} API 키를 입력하세요.`)
      return
    }
    setNotice(null)
    const targets = jobsRef.current.filter((j) => keys.includes(j.key))
    for (const j of targets) {
      patchJob(j.key, {
        status: 'uploading',
        phase: j.docId ? 'queued' : 'wait-upload',
        engine,
        model,
        error: null,
        done: 0,
        total: 0,
        lastPhase: null,
        queuePos: null,
        elapsed: null,
      })
    }
    const startOne = async (job: Job) => {
      try {
        let docId = job.docId
        if (!docId) {
          patchJob(job.key, { phase: 'uploading' })
          const file = filesRef.current.get(job.key)
          if (!file) throw new Error('원본 파일이 없습니다. 다시 추가해 주세요.')
          const doc = await uploadDocument(file)
          docId = doc.id
          filesRef.current.delete(job.key)
          patchJob(job.key, { docId, numPages: doc.num_pages })
        }
        await startTranslation(docId, { apiKey: key, apiModel, ollamaModel })
        patchJob(job.key, { status: 'translating', phase: 'queued' })
      } catch (e) {
        patchJob(job.key, {
          status: 'error',
          phase: 'error',
          error: e instanceof Error ? e.message : '번역을 시작하지 못했습니다.',
        })
      }
    }
    const pending = [...targets]
    await Promise.all(
      Array.from({ length: Math.min(UPLOAD_CONCURRENCY, pending.length) }, async () => {
        while (pending.length) await startOne(pending.shift()!)
      }),
    )
  }

  const cancelJob = async (job: Job) => {
    patchJob(job.key, { cancelling: true })
    try {
      await cancelTranslation(job.docId!)
    } catch (e) {
      patchJob(job.key, { cancelling: false })
      setNotice(e instanceof Error ? e.message : '취소하지 못했습니다.')
    }
  }

  const removeJob = (key: string) => {
    filesRef.current.delete(key)
    setJobs((list) => list.filter((j) => j.key !== key))
  }

  const handleDrop = (e: DragEvent<HTMLDivElement>) => {
    e.preventDefault()
    setDragOver(false)
    const hasFolder = Array.from(e.dataTransfer.items ?? []).some(
      (item) => item.webkitGetAsEntry()?.isDirectory,
    )
    void readDropped(e.dataTransfer).then((picked) => addFiles(picked, hasFolder))
  }

  const drafts = jobs.filter((j) => j.status === 'draft')
  const count = (pred: (j: Job) => boolean) => jobs.filter(pred).length
  const nDone = count((j) => j.status === 'done')
  const nActive = count(isActive)
  const nProblem = count(isFailed)
  const started = jobs.filter((j) => j.status !== 'draft')
  const overall = started.length
    ? Math.round(started.reduce((s, j) => s + jobPercent(j), 0) / started.length)
    : 0
  const hasBoard = jobs.length > 0

  return (
    <div className="page">
      <header className="upload-header app-header">
        <span className="logo">paperTranslate</span>
        <div className="settings-bar">
          <select
            className="model-select"
            aria-label="번역 모델"
            value={model}
            onChange={(e) => setModel(e.target.value)}
          >
            {PROVIDER_ORDER.map((p) => (
              <optgroup key={p} label={PROVIDER_GROUP[p]}>
                {API_MODELS.filter((m) => m.provider === p).map((m) => (
                  <option key={m.id} value={m.id}>
                    {m.label}
                  </option>
                ))}
              </optgroup>
            ))}
            <optgroup label="로컬 (이 PC의 Ollama)">
              {localModels?.models.map((m) => (
                <option key={m.name} value={LOCAL_PREFIX + m.name}>
                  {m.name}
                  {m.size_gb ? ` · ${m.size_gb}GB` : ''}
                </option>
              ))}
              {/* Keep the saved choice selectable even if it is not listed. */}
              {isLocal && !localModels?.models.some((m) => LOCAL_PREFIX + m.name === model) && (
                <option value={model}>{modelLabel(model)}</option>
              )}
              {localModels && !localModels.available && (
                <option disabled>Ollama가 실행 중이 아닙니다</option>
              )}
              {localModels?.available && localModels.models.length === 0 && (
                <option disabled>설치된 로컬 모델이 없습니다</option>
              )}
            </optgroup>
          </select>
          <div className={`key-field${isLocal ? ' key-field-off' : ''}`}>
            <input
              type={showKey ? 'text' : 'password'}
              autoComplete="off"
              spellCheck={false}
              placeholder={
                provider
                  ? `${PROVIDERS[provider].name} API 키 (${PROVIDERS[provider].keyHint})`
                  : '로컬 모델은 키가 필요 없습니다'
              }
              aria-label={provider ? `${PROVIDERS[provider].name} API 키` : 'API 키'}
              value={apiKey}
              disabled={isLocal}
              onChange={(e) => setApiKey(e.target.value)}
            />
            {apiKey && !isLocal && (
              <>
                <button type="button" onClick={() => setShowKey((v) => !v)}>
                  {showKey ? '숨김' : '보기'}
                </button>
                <button type="button" onClick={() => setApiKey('')} title="이 브라우저에서 키 삭제">
                  지우기
                </button>
              </>
            )}
          </div>
        </div>
        <span className="header-spacer" />
      </header>
      <main className={hasBoard ? 'batch-main' : 'upload-main'}>
        <div className={hasBoard ? 'batch-inner' : undefined}>
          <div
            className={`dropzone${hasBoard ? ' dropzone-compact' : ''}${dragOver ? ' dragover' : ''}`}
            onClick={() => inputRef.current?.click()}
            onDrop={handleDrop}
            onDragOver={(e) => {
              e.preventDefault()
              setDragOver(true)
            }}
            onDragLeave={() => setDragOver(false)}
            role="button"
            tabIndex={0}
            onKeyDown={(e) => {
              if (e.key === 'Enter' || e.key === ' ') inputRef.current?.click()
            }}
          >
            <p className="dropzone-title">PDF 파일이나 폴더를 끌어다 놓으세요</p>
            <div className="dropzone-actions">
              <button
                type="button"
                className="btn btn-sm"
                onClick={(e) => {
                  e.stopPropagation()
                  inputRef.current?.click()
                }}
              >
                파일 선택
              </button>
              <button
                type="button"
                className="btn btn-sm"
                onClick={(e) => {
                  e.stopPropagation()
                  folderInputRef.current?.click()
                }}
              >
                폴더 선택
              </button>
            </div>
          </div>
          <input
            ref={inputRef}
            type="file"
            accept="application/pdf,.pdf"
            multiple
            style={{ display: 'none' }}
            onChange={(e) => {
              addFiles(Array.from(e.target.files ?? []).map((file) => ({ file, path: file.name })))
              e.target.value = ''
            }}
          />
          <input
            ref={folderInputRef}
            type="file"
            multiple
            style={{ display: 'none' }}
            onChange={(e) => {
              const files = Array.from(e.target.files ?? [])
              addFiles(
                files.map((file) => ({ file, path: file.webkitRelativePath || file.name })),
                true,
              )
              e.target.value = ''
            }}
          />
          {notice && <div className="upload-error">{notice}</div>}

          {drafts.length > 0 && (
            <div className="batch-start">
              <span>{drafts.length}편 준비됨</span>
              <div className="batch-start-actions">
                <button type="button" className="btn" onClick={() => drafts.forEach((j) => removeJob(j.key))}>
                  비우기
                </button>
                <button
                  type="button"
                  className="btn btn-accent"
                  onClick={() => void run(drafts.map((j) => j.key))}
                >
                  {drafts.length}편 번역 시작
                </button>
              </div>
            </div>
          )}

          {hasBoard && (
            <>
              {/* Overall progress: just the gauge, plus the two bulk actions. */}
              <div className="batch-summary">
                <div className="progress-track batch-track" aria-label="전체 진행률">
                  <div className="progress-fill" style={{ width: `${overall}%` }} />
                </div>
                {(nActive > 0 || nDone + nProblem > 0) && (
                  <div className="batch-summary-actions">
                    {nActive > 0 && (
                      <button
                        type="button"
                        className="btn btn-sm"
                        onClick={() =>
                          jobs
                            .filter((j) => j.docId && j.status === 'translating' && !j.cancelling)
                            .forEach((j) => void cancelJob(j))
                        }
                      >
                        모두 취소
                      </button>
                    )}
                    {nDone + nProblem > 0 && (
                      <button
                        type="button"
                        className="btn btn-sm"
                        onClick={() =>
                          setJobs((list) =>
                            list.filter((j) => !(j.status === 'done' || isFailed(j))),
                          )
                        }
                      >
                        끝난 항목 정리
                      </button>
                    )}
                  </div>
                )}
              </div>

              <div className="job-grid">
                {jobs.map((job) => (
                  <JobCard
                    key={job.key}
                    job={job}
                    onOpen={() =>
                      onOpen({
                        id: job.docId!,
                        filename: job.filename,
                        num_pages: job.numPages ?? 0,
                        status: 'done',
                      })
                    }
                    onRetry={() => void run([job.key])}
                  />
                ))}
              </div>
            </>
          )}
        </div>
      </main>
    </div>
  )
}

export default BatchPage
