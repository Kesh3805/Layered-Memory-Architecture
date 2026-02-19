#!/bin/bash
echo "Starting RAG Chat App..."
echo ""
echo "The first startup may take a moment to download the embedding model (~80MB)"
echo ""
cd "$(dirname "$0")"
.venv/bin/python -m uvicorn main:app --reload --host 0.0.0.0
