# Builds the always-on web app only (main.py: routes, uploads, status
# polling, and the live "Ask This Lecture" endpoint). It installs the SLIM
# requirements.txt, which excludes crewai/pymupdf/pytesseract/pillow/
# edge-tts — none of that runs in this container anymore. The actual
# six-stage pipeline runs separately, in a GitHub Actions job (see
# .github/workflows/process_document.yml and worker.py), which is why no
# system packages (Tesseract, ffmpeg) are needed here either.
#
# To run the full pipeline locally instead (PIPELINE_EXECUTOR=thread, the
# default for local dev), use requirements-worker.txt directly with a
# plain `pip install`, not this Dockerfile.
FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000}"]
