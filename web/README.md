# Web — Start Guide

The web app consists of two processes that must both be running:

| Process | Stack | Port |
|---------|-------|------|
| **Backend** | FastAPI + uvicorn | `8000` |
| **Frontend** | Vite + React | `5173` |

The Vite dev server proxies `/infer` and `/health` to `localhost:8000`, so there is no CORS issue during development.

---

## Backend

Run from the **project root**:

```bash
uv run uvicorn web.backend.main:app --reload --port 8000
```

The first run loads all three model weights from `weight/`. Expect a few seconds on CPU before you see:

```
Models ready.
INFO:     Uvicorn running on http://0.0.0.0:8000
```

---

## Frontend

Run from the **project root**:

```bash
# install dependencies (first time only)
npm install --prefix web/frontend

# start dev server
npm run dev --prefix web/frontend
```

Then open **http://localhost:5173** in your browser.

---

## Both at once (two terminals)

**Terminal 1 — backend:**
```bash
uv run uvicorn web.backend.main:app --reload --port 8000
```

**Terminal 2 — frontend:**
```bash
npm run dev --prefix web/frontend
```

---

## Production build

```bash
# build frontend static files
npm run build --prefix web/frontend

# serve backend (serves API only; host the dist/ folder separately or via nginx)
uv run uvicorn web.backend.main:app --host 0.0.0.0 --port 8000
```

Built files land in `web/frontend/dist/`.
