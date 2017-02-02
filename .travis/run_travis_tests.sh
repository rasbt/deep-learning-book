#!/usr/bin/env bash

set -e

if [[ "$NOTEBOOKS" == "true" ]]; then
    python -m unittest discover code -v
else
    python -m unittest discover ann -v
fi

if [[ "$DOCTESTS" == "true" ]]; then
    python -m doctest ann/np/preprocessing.py  -v
    python -m doctest ann/np/scoring.py -v
    python -m doctest ann/np/training.py -v
    python -m doctest ann/np/lossfunctions.py -v
    python -m doctest ann/np/activations.py -v
fi
