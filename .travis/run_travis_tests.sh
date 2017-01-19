#!/usr/bin/env bash

set -e

if [[ "$NOTEBOOKS" == "true" ]]; then
    python -m unittest discover code -v
else
    python -m unittest discover ann -v
fi
