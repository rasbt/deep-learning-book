#!/usr/bin/env bash

set -e

if [[ "$NOTEBOOKS" == "true" ]]; then
    python -m unittest discover code -v
else
    python -m unittest discover ann -v
fi

if

python -m doctest ann/np/preprocessing.py  -v
python -m doctest ann/np/scoring.py -v
python -m doctest ann/np/training.py -v
