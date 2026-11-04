# Advanced Version — Audit & Product Build-Out

This document records the professional-presentation audit performed on top
of the existing four-week churn-prediction project, and everything added on
top of it. The underlying ML work (v1/v2 models, stacking ensemble,
threshold optimization, permutation importance, calibration, precision-recall
analysis, automated model tests, CI, Docker) was already reasonably
comprehensive going into this pass — the gaps were in product presentation,
API/frontend integration, and a few cross-file consistency issues, not
missing ML work.

## What the audit found

1. **Streamlit v2 risk logic was inconsistent with the documented optimized
   threshold.** `app.py` and `Week3_ML_Model_Development/serving_api.py` did
   not read the same decision threshold as `MODEL_CARD.md` and
   `metrics_v2.json`, so the "High risk" cutoff shown to a user in the demo
   app could silently disagree with the number the project's own
   documentation claimed.
2. **The FastAPI backend (`serving_api.py`) was functional but minimal** —
   no CORS, no portfolio-level analytics endpoint, no static frontend to
   serve, and it always loaded the v1 model even though v2 was the better
   model per `MODEL_CARD.md`.
3. **Two different HTML dashboard implementations existed**
   (`Week4_Report_Presentation/outputs/dashboard.html` and
   `interactive_dashboard.html`) — useful as Week 4 assignment deliverables,
   but not a single professional product surface.
4. **API-level testing was missing.** `Week3_ML_Model_Development/tests/`
   covers the model pipeline well, but nothing exercised the serving layer
   itself (status codes, input validation, response shape, CORS).
5. **The synthetic-data caveat was correct but buried.** The project
   correctly discloses that the model is trained on synthetic data, but that
   warning wasn't prominent in the main-facing application surfaces.

## What was added

**`api.py`** (repo root) — the product-facing backend, superseding
`serving_api.py` for presentation purposes (that file stays in place
unchanged as the literal Week 3 assignment deliverable):
- Loads the v2 stacking-ensemble model by default, with a documented,
  logged fallback to v1 if v2 artifacts are missing.
- Reads the decision threshold (`0.206`) directly from `metrics_v2.json` —
  the same source of truth `app.py` and `MODEL_CARD.md` now use — instead
  of hardcoding a number that can drift out of sync.
- CORS enabled (permissive `*` origin for this demo/portfolio context —
  see "Recommended next stage" below for tightening it).
- `GET /portfolio` — dataset-level KPIs, revenue-at-risk, and churn-rate
  breakdowns by plan/contract/region/engagement, so the frontend never
  needs direct file access to the dataset.
- Mounts `static/` so `uvicorn api:app` alone serves both the API and the
  control-center UI at `/`, with Swagger docs at `/docs`.

**`static/index.html`** — the "Retention Console" control center, as a
single self-contained file (HTML + CSS in one `<style>` block + JS in one
`<script>` block — no separate files, no build step), replacing the two
standalone dashboard HTML files as the primary presentation surface.
Sidebar navigation across four panels (Overview, Risk Predictor, Model
Health, About), portfolio KPI cards, a churn-exposure bar chart with
segment tabs, a revenue-at-risk donut, a live customer risk predictor
backed by `POST /predict`, an SVG instrument-style risk gauge with the
model's *actual* threshold marked on the dial (not a generic 50% split),
an API-health indicator polling `/health`, and a persistent synthetic-data
warning banner. Down to ~720px the sidebar collapses to a horizontal bar
and grids stack to one column. Every CSS/JS block is commented by section
(design tokens, sidebar, gauge math, boot sequence, etc.) so the whole
file reads top to bottom without needing to jump between files.

**Backend hardening (`api.py`):** startup model-preload via a `lifespan`
handler (so a missing model file fails loudly at server start, not on a
random first request), request-logging middleware (every call logged as
`METHOD /path -> status (Nms)`), and a catch-all exception handler that
turns any unhandled error into a clean `500` JSON response instead of a
raw traceback or a dropped connection.

**`tests/test_api.py`** — a pytest suite against `api.py` via FastAPI's
`TestClient` (14 tests, all passing): health check, model-info shape and
threshold-consistency (this test would have caught issue #1 above),
predict validation (missing field, out-of-range value, invalid category
→ all `422`), single-vs-batch prediction agreement, batch empty-list
rejection, portfolio KPI range checks, and a CORS-header check. Tests
that depend on the trained model or dataset skip (rather than fail) with
a clear reason if those artifacts aren't present, so CI failures stay
diagnostic.

**Fixes to existing files:**
- `app.py` (Streamlit) — now loads `DECISION_THRESHOLD` from
  `metrics_v2.json` and uses it for the risk-band split, instead of a
  cutoff that could disagree with `MODEL_CARD.md`.
- `requirements-advanced.txt` — added `pytest` and `httpx` for the new
  test suite.

## Verified working

- All 12 existing tests in `Week3_ML_Model_Development/tests/` still pass.
- All 14 new tests in `tests/test_api.py` pass against the real
  `churn_model_v2.joblib` and `churn_dataset.csv`.
- `uvicorn api:app` was run locally and every route was hit with `curl`:
  `/` (the control center), `/docs`, `/health`, `/model-info`, `/predict`,
  `/predict/batch`, and `/portfolio` all returned the expected status
  codes and payload shapes, including `422` on invalid input and `400` on
  an empty batch request. The startup log confirmed the model preloads
  correctly and every request is logged with its status and timing.

## Recommended next stage

If the goal is to push this toward a real final-year / portfolio /
production-style project, in roughly the order I'd tackle them:

1. **Authentication + role-based access** — the API is currently
   unauthenticated by design (portfolio demo); add API keys or OAuth
   before any real deployment.
2. **A database for prediction history** — right now nothing persists;
   even SQLite would let the console show a real audit trail.
3. **Customer search/history page** — look up a specific customer's past
   scores over time, not just one-off predictions.
4. **Batch CSV upload** in the UI (the API already supports
   `/predict/batch`; the frontend doesn't expose a file-upload path to it
   yet).
5. **Prediction audit log** — who scored what, when, with what result.
6. **Model version selector** in the UI — let a user compare v1 vs v2
   side by side instead of always serving whichever is "best."
7. **Drift monitoring** — compare live scoring-time feature distributions
   against the training distribution.
8. **Calibration monitoring over time** — `advanced_interpretability.py`
   already produces a one-time calibration curve; track it per retrain.
9. **Business-cost-based threshold optimization** instead of F1 alone —
   weight false negatives (a missed churner) against false positives (a
   wasted retention email) using an actual estimated cost ratio.
10. **Broaden `tests/test_api.py`** with load/concurrency tests and a CI
    job that runs it automatically (the existing `.github/workflows/ci.yml`
    currently only runs the model-pipeline tests).
11. **End-to-end browser tests** (e.g., Playwright) against the static
    control center, not just the API layer.
12. **Tighten CORS** to a specific origin allowlist instead of `*` before
    any real deployment.
13. **A real model registry/versioning scheme** instead of two hardcoded
    `.joblib` paths.
14. **Real-data temporal validation** — everything here is validated on a
    single synthetic-data split; a production system needs time-based
    validation on real, evolving data.
15. **Deployment configuration** for actual cloud hosting (the `Dockerfile`
    and `docker-compose.yml` already exist as a starting point).

None of the above are required for this to be a strong, honest portfolio
submission as-is — they're what would separate a strong student project
from something you'd actually trust with real subscriber data and real
retention budget.
