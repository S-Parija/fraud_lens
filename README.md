# FraudLens

Bank transaction screening with HTML, CSS, JavaScript, and FastAPI. Trained on synthetic transaction data.

## Setup

Requires Python 3.12.

```bash
git clone https://github.com/S-Parija/fraud_lens.git
cd fraud_lens
python -m venv .venv
```

Activate the environment:

- Linux/macOS: `source .venv/bin/activate`
- Windows Command Prompt: `.venv\Scripts\activate.bat`

```bash
python -m pip install -r requirements.txt
python main.py
```

Open http://127.0.0.1:8000. The server serves both the page and prediction API.

## Development

```bash
python -m uvicorn backend.app:app --reload
```

The frontend files are `frontend/index.html`, `frontend/static/styles.css`, and `frontend/static/app.js`. Edit them and refresh the browser.

## Tests

```bash
python -m pytest -q
```

## Render

- Root Directory: leave empty
- Build Command: `pip install -r requirements.txt`
- Start Command: `python -m uvicorn backend.app:app --host 0.0.0.0 --port $PORT`
- Health Check Path: `/api/health`

The repository specifies Python 3.12 in `.python-version`.
