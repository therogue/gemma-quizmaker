#!/usr/bin/env bash
set -euo pipefail
uv run uvicorn app.main:app \
  --reload \
  --reload-dir app \
  --reload-dir quizmaker \
  --port 8000
