"""Start the application with `python main.py`."""

import uvicorn


def main():
    """Start the local API (and the frontend if it has been built)."""
    uvicorn.run("backend.app:app", host="127.0.0.1", port=8000)


if __name__ == "__main__":
    main()
