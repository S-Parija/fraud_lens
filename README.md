# FraudLens

Transaction checking app built with React and FastAPI. Includes the trained model and sample transactions. Uses synthetic data for demonstration.

## Requirements

- Python 3.12 and [UV](https://docs.astral.sh/uv/)
- Node.js 22.12 or later

## Run

```bash
git clone https://github.com/S-Parija/fraud_lens.git
cd fraud_lens
uv sync --locked
cd frontend
npm ci
npm run build
cd ..
uv run python main.py
```

Open http://127.0.0.1:8000. Stop with Ctrl+C.

On Windows, `./run.ps1` also installs dependencies, builds the frontend, and starts the app. It can set up portable Node.js if needed.

## Development

Start the backend from the repository root:

```bash
uv run uvicorn backend.app:app --reload --host 127.0.0.1 --port 8000
```

In a second terminal:

```bash
cd frontend
npm run dev
```

Open http://127.0.0.1:5173. API documentation is available at http://127.0.0.1:8000/docs.

## Tests

```bash
uv run pytest -q
cd frontend
npm test
```

The saved `.pkl` file is required at runtime. The native binary-model JSON and example prediction JSON are used by the tests. Load only trusted model files.
