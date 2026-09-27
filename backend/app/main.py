"""FastAPI application entry point.

Run from backend/:  .venv/Scripts/python.exe -m uvicorn app.main:app --port 8000
"""

from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app import __version__, config, db
from app.api.app_routes import router as app_router
from app.api.routes import router


@asynccontextmanager
async def lifespan(_app: FastAPI):
    config.ensure_dirs()
    db.init_db()
    # Recovery after an unclean shutdown: a document left in 'translating'
    # has no running pipeline task in this process, so surface it as error.
    db.mark_interrupted_translations()
    yield


app = FastAPI(title="paperTranslate", version=__version__, lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    # Only the Vite dev server (npm run dev on :5173) is a separate origin;
    # the built frontend is served by this app itself (same origin).
    allow_origin_regex=r"http://(localhost|127\.0\.0\.1):5173",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router, prefix="/api")
app.include_router(app_router, prefix="/api/app")

# Serve the frontend build (frontend/dist, or PAPERTRANSLATE_FRONTEND_DIST in
# the packaged desktop app) if it exists.
_DIST_DIR = config.get_frontend_dist()
if _DIST_DIR.is_dir() and (_DIST_DIR / "index.html").is_file():
    _assets = _DIST_DIR / "assets"
    if _assets.is_dir():
        app.mount("/assets", StaticFiles(directory=str(_assets)), name="assets")

    @app.get("/{full_path:path}", include_in_schema=False)
    async def spa_fallback(full_path: str):
        # API routes are matched first by FastAPI; this only catches non-API paths.
        if full_path.startswith("api/"):
            raise HTTPException(status_code=404, detail="찾을 수 없습니다")
        candidate = (_DIST_DIR / full_path).resolve()
        if (
            full_path
            and candidate.is_file()
            and str(candidate).startswith(str(_DIST_DIR))
        ):
            return FileResponse(str(candidate))
        return FileResponse(str(_DIST_DIR / "index.html"))
