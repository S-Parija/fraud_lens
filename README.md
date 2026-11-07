# FraudLens

Banking transaction risk assessment with React, FastAPI, and XGBoost. Trained on synthetic transaction data.

## Setup

Requires Python 3.12 and Node.js 22.12 or later.

```bash
git clone https://github.com/S-Parija/fraud_lens.git
cd fraud_lens
python -m venv .venv
```

Activate the environment:

- Linux/macOS: `source .venv/bin/activate`
- Windows Command Prompt: `.venv\Scripts\activate.bat`

Install dependencies, build the frontend, and start the app:

```bash
python -m pip install -r requirements.txt
cd frontend
npm ci
npm run build
cd ..
python main.py
```

Open http://127.0.0.1:8000.

## Development

Run the backend:

```bash
python -m uvicorn backend.app:app --reload
```

In a separate terminal:

```bash
cd frontend
npm run dev
```

Open http://127.0.0.1:5173. API docs: http://127.0.0.1:8000/docs.

## Tests

```bash
python -m pytest -q
cd frontend
npm test
```
