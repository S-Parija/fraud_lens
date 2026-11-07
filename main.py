"""Start the application with `python main.py`."""

import uvicorn


def main():
    """Serve the transaction form and prediction API."""
    uvicorn.run("backend.app:app", host="127.0.0.1", port=8000)


if __name__ == "__main__":
    main()
