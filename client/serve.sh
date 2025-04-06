#!/usr/bin/env bash

if [ -z "$(command -v python3)" ]; then
    echo "I don't have python3. Sorry, run the server another way.";
    exit 1;
fi

python3 -m http.server
