#!/usr/bin/env bash

if [ -z "$(command -v poetry)" ]; then
    echo "poetry isn't available. Why not?"
    exit 1
fi

if [ -z "$(poetry show | grep uvicorn)" ]; then
    echo "No uvicorn in the poetry environment. Wat?"
    exit 1
fi

poetry run uvicorn \
    --reload \
    --host 0.0.0.0 \
    --port 9090 \
    --app-dir src/server/ main:app
